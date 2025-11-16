// Function: sub_2B1BC20
// Address: 0x2b1bc20
//
char __fastcall sub_2B1BC20(__int64 **a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r14
  char result; // al
  __int64 v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned int v16; // eax
  unsigned int v17; // esi
  __int64 v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rdx
  unsigned int v21; // eax
  __int64 v22; // rax
  char v23; // dl
  char v24; // al
  char v25; // al
  char *v26; // r14
  char *v27; // r8
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // rsi
  __int64 v31; // rdx
  __int64 *v32; // rax
  __int64 v33; // rsi
  char v34; // cl
  bool v35; // al
  unsigned __int64 v36; // rcx
  char *v37; // rdx
  char v38; // di
  unsigned __int8 v39; // si
  __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // r8
  __int64 v43; // rax
  char *v44; // rdi
  __int64 v45; // rax
  unsigned int v46; // [rsp+Ch] [rbp-44h]
  unsigned int v47; // [rsp+Ch] [rbp-44h]
  __int64 v48; // [rsp+10h] [rbp-40h]
  __int64 v49; // [rsp+18h] [rbp-38h]

  v3 = a3;
  v4 = a2;
  v5 = **a1;
  v6 = *(_QWORD *)(v5 + 8LL * a2);
  v7 = *(_QWORD *)(v5 + 8LL * a3);
  if ( v6 == v7 || !(unsigned int)sub_BD3960(*(_QWORD *)(v5 + 8LL * a2)) && !(unsigned int)sub_BD3960(v7) )
    return 0;
  if ( *(_BYTE *)v6 == 13 )
    return 1;
  if ( *(_BYTE *)v7 == 13 )
    return 0;
  v46 = sub_BD3960(v6);
  if ( v46 < (unsigned int)sub_BD3960(v7) )
    return 1;
  v47 = sub_BD3960(v6);
  if ( v47 > (unsigned int)sub_BD3960(v7) )
    return 0;
  v10 = *(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL);
  v11 = *(_QWORD *)(*(_QWORD *)(v7 + 16) + 24LL);
  v12 = *(_QWORD *)(v10 + 40);
  v13 = *(_QWORD *)(v11 + 40);
  if ( v12 != v13 )
  {
    v14 = *(_QWORD *)(*a1[1] + 3320);
    if ( v12 )
    {
      v15 = (unsigned int)(*(_DWORD *)(v12 + 44) + 1);
      v16 = *(_DWORD *)(v12 + 44) + 1;
    }
    else
    {
      v15 = 0;
      v16 = 0;
    }
    v17 = *(_DWORD *)(v14 + 32);
    if ( v16 >= v17 )
      return 0;
    v18 = *(_QWORD *)(v14 + 24);
    v19 = *(_QWORD *)(v18 + 8 * v15);
    if ( !v19 )
      return 0;
    if ( v13 )
    {
      v20 = (unsigned int)(*(_DWORD *)(v13 + 44) + 1);
      v21 = v20;
    }
    else
    {
      v20 = 0;
      v21 = 0;
    }
    if ( v17 > v21 )
    {
      v22 = *(_QWORD *)(v18 + 8 * v20);
      if ( v22 )
        return *(_DWORD *)(v19 + 72) < *(_DWORD *)(v22 + 72);
    }
    return 1;
  }
  v23 = *(_BYTE *)v10;
  v24 = *(_BYTE *)v11;
  if ( *(_BYTE *)v10 == 91 )
  {
    if ( v24 == 91 )
    {
      v25 = 1;
      goto LABEL_25;
    }
    v34 = 1;
    v10 = 0;
LABEL_40:
    if ( v24 != 90 )
      v11 = 0;
    v27 = (char *)v11;
    if ( v34 )
      return 1;
    goto LABEL_43;
  }
  if ( v24 != 91 )
  {
    v34 = 0;
    if ( v23 != 90 )
      v10 = 0;
    goto LABEL_40;
  }
  if ( v23 == 90 )
  {
    v26 = 0;
    v25 = 0;
    goto LABEL_26;
  }
  v25 = 0;
  v10 = 0;
LABEL_25:
  v26 = (char *)v10;
  v10 = 0;
LABEL_26:
  if ( !v26 && v11 )
    return 0;
  v27 = 0;
  if ( !v11 || !v25 )
  {
LABEL_43:
    if ( !v27 && v10 )
      return 1;
    v35 = v27 != 0 && v10 == 0;
    if ( !v35 && v10 && v27 )
    {
      v36 = *(_QWORD *)(v10 - 64);
      v37 = (char *)*((_QWORD *)v27 - 8);
      v38 = *(_BYTE *)v36;
      v39 = *v37;
      if ( *(_BYTE *)v36 <= 0x1Cu )
      {
        if ( v39 > 0x1Cu )
        {
          v41 = 0;
          if ( v38 == 22 )
            v41 = *(_QWORD *)(v10 - 64);
          v40 = 0;
LABEL_51:
          if ( (char *)v36 != v37 )
          {
            v35 = v37 != 0;
            if ( !v40 && v37 )
              return 0;
            v42 = *((_QWORD *)v27 - 8);
            v37 = 0;
            goto LABEL_55;
          }
          goto LABEL_83;
        }
        if ( v38 == 22 )
        {
          if ( v39 == 22 )
          {
            if ( v37 != (char *)v36 )
            {
              v41 = *(_QWORD *)(v10 - 64);
              v42 = 0;
              v40 = 0;
LABEL_55:
              if ( v40 && v35 )
              {
                v33 = *(_QWORD *)(v40 + 40);
                v31 = *(_QWORD *)(v42 + 40);
                if ( v31 != v33 )
                {
                  v32 = a1[1];
                  return sub_2B12300(*v32, v33, v31);
                }
                v30 = v42;
                return sub_B445A0(v40, v30);
              }
LABEL_80:
              if ( v41 || !v37 )
                return *((_DWORD *)v37 + 8) > *(_DWORD *)(v41 + 32);
              return 0;
            }
LABEL_83:
            v44 = (char *)v10;
            v49 = sub_2B18C70(v27, 0);
            goto LABEL_63;
          }
          v45 = 0;
          return (v45 | v36) != 0;
        }
        v45 = 0;
      }
      else
      {
        if ( v39 > 0x1Cu )
        {
          v40 = *(_QWORD *)(v10 - 64);
          v41 = 0;
          goto LABEL_51;
        }
        v45 = *(_QWORD *)(v10 - 64);
      }
      if ( v39 == 22 )
      {
        if ( (char *)v36 != v37 )
        {
          v41 = 0;
          goto LABEL_80;
        }
        goto LABEL_83;
      }
      v36 = 0;
      return (v45 | v36) != 0;
    }
    return 0;
  }
  v28 = *a1[2];
  v29 = *(_QWORD *)(v28 + 8LL * a2);
  if ( !v29 )
    return 0;
  v30 = *(_QWORD *)(v28 + 8 * v3);
  if ( !v30 )
    return 1;
  if ( v29 != v30 )
  {
    v31 = *(_QWORD *)(v30 + 40);
    if ( v31 != *(_QWORD *)(v29 + 40) )
    {
      v32 = a1[1];
      v33 = *(_QWORD *)(v29 + 40);
      return sub_2B12300(*v32, v33, v31);
    }
    v40 = *(_QWORD *)(v28 + 8 * v4);
    return sub_B445A0(v40, v30);
  }
  v43 = sub_2B18C70((char *)v11, 0);
  v44 = v26;
  v49 = v43;
LABEL_63:
  v48 = sub_2B18C70(v44, 0);
  result = BYTE4(v49);
  if ( BYTE4(v49) && BYTE4(v48) )
    return (unsigned int)v48 < (unsigned int)v49;
  return result;
}

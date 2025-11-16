// Function: sub_10115A0
// Address: 0x10115a0
//
__int64 __fastcall sub_10115A0(__int64 a1, _BYTE *a2, __int64 a3, __m128i *a4, int a5)
{
  __int64 v5; // r12
  unsigned __int8 v7; // bl
  unsigned int v8; // eax
  _BYTE *v9; // rdx
  unsigned int v10; // r14d
  __int64 *v11; // r13
  unsigned __int64 v12; // rdi
  __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  __int64 v15; // r8
  int v16; // edx
  char v18; // al
  __int64 v19; // r8
  unsigned int v20; // r12d
  __int64 v21; // rdi
  int v22; // eax
  bool v23; // al
  char v24; // al
  unsigned int v25; // r12d
  bool v26; // al
  __int64 v27; // rax
  char v28; // al
  __int64 *v29; // rax
  __int64 v31; // r12
  __int64 v32; // rdx
  _BYTE *v33; // rax
  __int64 v34; // r12
  __int64 v35; // rdx
  _BYTE *v36; // rax
  unsigned __int8 *v37; // r8
  unsigned int v38; // r12d
  bool v39; // r12
  unsigned int v40; // r14d
  __int64 v41; // rax
  unsigned int v42; // r12d
  int v43; // eax
  unsigned __int8 *v44; // rbx
  unsigned int v45; // r12d
  bool v46; // r14
  __int64 v47; // rax
  unsigned int v48; // r14d
  __int64 v49; // [rsp+0h] [rbp-60h]
  __int64 v50; // [rsp+10h] [rbp-50h]
  __int64 v51; // [rsp+18h] [rbp-48h]
  int v52; // [rsp+18h] [rbp-48h]
  __int64 v53; // [rsp+18h] [rbp-48h]
  __int64 v55; // [rsp+20h] [rbp-40h]
  __int64 v56; // [rsp+20h] [rbp-40h]
  __int64 v57; // [rsp+20h] [rbp-40h]
  __int64 v58; // [rsp+20h] [rbp-40h]
  __int64 *v59; // [rsp+20h] [rbp-40h]
  __int64 v60; // [rsp+20h] [rbp-40h]
  __int64 v61; // [rsp+20h] [rbp-40h]
  int v62; // [rsp+20h] [rbp-40h]
  unsigned int v63; // [rsp+2Ch] [rbp-34h]

  v63 = a5 - 1;
  if ( !a5 )
    return 0;
  v5 = (__int64)a2;
  if ( *a2 == 86 )
  {
    v10 = a1;
    v9 = a2;
    v7 = BYTE4(a1);
    v5 = a3;
  }
  else
  {
    v7 = 0;
    v8 = sub_B52F50(a1);
    v9 = (_BYTE *)a3;
    v10 = v8;
  }
  v11 = (__int64 *)*((_QWORD *)v9 - 12);
  v51 = *((_QWORD *)v9 - 4);
  v55 = *((_QWORD *)v9 - 8);
  v49 = sub_AD6400(v11[1]);
  v50 = v7;
  v12 = ((unsigned __int64)v7 << 32) | v10;
  if ( v10 - 32 <= 9 )
  {
    v13 = sub_1012FB0(v12, v55, v5, a4);
    if ( v11 == (__int64 *)v13 )
      goto LABEL_17;
  }
  else
  {
    v13 = sub_1011B90(v12, v55, v5, 0, a4, v63);
    if ( v11 == (__int64 *)v13 )
      goto LABEL_17;
  }
  if ( v13 )
    goto LABEL_7;
  if ( !sub_FFED40((__int64)v11, v10, v55, v5) )
    return 0;
LABEL_17:
  v13 = v49;
  if ( !v49 )
    return 0;
LABEL_7:
  v56 = sub_AD6450(v11[1]);
  v14 = v10 | (unsigned __int64)(v50 << 32);
  if ( v10 - 32 > 9 )
  {
    v15 = sub_1011B90(v14, v51, v5, 0, a4, v63);
    if ( v11 == (__int64 *)v15 )
      goto LABEL_33;
  }
  else
  {
    v15 = sub_1012FB0(v14, v51, v5, a4);
    if ( v11 == (__int64 *)v15 )
      goto LABEL_33;
  }
  if ( v15 )
    goto LABEL_10;
  if ( !sub_FFED40((__int64)v11, v10, v51, v5) )
    return 0;
LABEL_33:
  if ( !v56 )
    return 0;
  v15 = v56;
LABEL_10:
  if ( v15 == v13 )
    return v13;
  v16 = *(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v11[1] + 8) - 17 > 1 )
  {
    if ( v16 == 18 || v16 == 17 )
      return 0;
  }
  else if ( v16 != 18 && v16 != 17 )
  {
    return 0;
  }
  v57 = v15;
  v18 = sub_FFFE90(v15);
  v19 = v57;
  if ( v18 )
  {
    v28 = sub_98EF70(v13, (__int64)v11);
    v19 = v57;
    if ( v28 )
    {
      v29 = (__int64 *)sub_101D750(v11, v13, a4, v63);
      v19 = v57;
      if ( v29 )
        return (__int64)v29;
    }
  }
  if ( *(_BYTE *)v13 == 17 )
  {
    v20 = *(_DWORD *)(v13 + 32);
    if ( v20 > 0x40 )
    {
      v58 = v19;
      v21 = v13 + 24;
LABEL_23:
      v22 = sub_C444A0(v21);
      v19 = v58;
      v23 = v20 - 1 == v22;
      goto LABEL_24;
    }
    v23 = *(_QWORD *)(v13 + 24) == 1;
  }
  else
  {
    v31 = *(_QWORD *)(v13 + 8);
    v32 = (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17;
    if ( (unsigned int)v32 > 1 || *(_BYTE *)v13 > 0x15u )
      goto LABEL_26;
    v58 = v19;
    v33 = sub_AD7630(v13, 0, v32);
    v19 = v58;
    if ( !v33 || *v33 != 17 )
    {
      if ( *(_BYTE *)(v31 + 8) == 17 )
      {
        v52 = *(_DWORD *)(v31 + 32);
        if ( v52 )
        {
          v39 = 0;
          v40 = 0;
          while ( 1 )
          {
            v61 = v19;
            v41 = sub_AD69F0((unsigned __int8 *)v13, v40);
            v19 = v61;
            if ( !v41 )
              break;
            if ( *(_BYTE *)v41 != 13 )
            {
              if ( *(_BYTE *)v41 != 17 )
                break;
              v42 = *(_DWORD *)(v41 + 32);
              if ( v42 <= 0x40 )
              {
                v39 = *(_QWORD *)(v41 + 24) == 1;
              }
              else
              {
                v43 = sub_C444A0(v41 + 24);
                v19 = v61;
                v39 = v42 - 1 == v43;
              }
              if ( !v39 )
                break;
            }
            if ( v52 == ++v40 )
            {
              if ( v39 )
                goto LABEL_25;
              goto LABEL_26;
            }
          }
        }
      }
      goto LABEL_26;
    }
    v20 = *((_DWORD *)v33 + 8);
    if ( v20 > 0x40 )
    {
      v21 = (__int64)(v33 + 24);
      goto LABEL_23;
    }
    v23 = *((_QWORD *)v33 + 3) == 1;
  }
LABEL_24:
  if ( v23 )
  {
LABEL_25:
    v59 = (__int64 *)v19;
    v24 = sub_98EF70(v19, (__int64)v11);
    v19 = (__int64)v59;
    if ( v24 )
    {
      v29 = sub_1010B00(v11, v59, a4, v63);
      if ( v29 )
        return (__int64)v29;
      v19 = (__int64)v59;
    }
  }
LABEL_26:
  if ( *(_BYTE *)v19 == 17 )
  {
    v25 = *(_DWORD *)(v19 + 32);
    if ( v25 <= 0x40 )
      v26 = *(_QWORD *)(v19 + 24) == 1;
    else
      v26 = v25 - 1 == (unsigned int)sub_C444A0(v19 + 24);
  }
  else
  {
    v34 = *(_QWORD *)(v19 + 8);
    v35 = (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17;
    if ( (unsigned int)v35 > 1 || *(_BYTE *)v19 > 0x15u )
      return 0;
    v60 = v19;
    v36 = sub_AD7630(v19, 0, v35);
    v37 = (unsigned __int8 *)v60;
    if ( !v36 || *v36 != 17 )
    {
      if ( *(_BYTE *)(v34 + 8) == 17 )
      {
        v62 = *(_DWORD *)(v34 + 32);
        if ( v62 )
        {
          v53 = v13;
          v44 = v37;
          v45 = 0;
          v46 = 0;
          while ( 1 )
          {
            v47 = sub_AD69F0(v44, v45);
            if ( !v47 )
              break;
            if ( *(_BYTE *)v47 != 13 )
            {
              if ( *(_BYTE *)v47 != 17 )
                break;
              v48 = *(_DWORD *)(v47 + 32);
              v46 = v48 <= 0x40 ? *(_QWORD *)(v47 + 24) == 1 : v48 - 1 == (unsigned int)sub_C444A0(v47 + 24);
              if ( !v46 )
                break;
            }
            if ( v62 == ++v45 )
            {
              v13 = v53;
              if ( v46 )
                goto LABEL_30;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v38 = *((_DWORD *)v36 + 8);
    if ( v38 <= 0x40 )
    {
      if ( *((_QWORD *)v36 + 3) != 1 )
        return 0;
      goto LABEL_30;
    }
    v26 = v38 - 1 == (unsigned int)sub_C444A0((__int64)(v36 + 24));
  }
  if ( !v26 )
    return 0;
LABEL_30:
  if ( !(unsigned __int8)sub_FFFE90(v13) )
    return 0;
  v27 = sub_AD62B0(v11[1]);
  return sub_101B6D0(v11, v27, a4, v63);
}

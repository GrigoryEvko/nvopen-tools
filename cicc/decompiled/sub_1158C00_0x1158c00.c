// Function: sub_1158C00
// Address: 0x1158c00
//
bool __fastcall sub_1158C00(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r12
  _BYTE *v3; // r12
  _BYTE *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 v9; // rax
  __int64 v10; // r14
  unsigned int v11; // r12d
  bool v12; // al
  __int64 *v13; // rax
  _BYTE *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  unsigned int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // r14
  unsigned int v20; // r12d
  __int64 *v21; // rax
  __int64 v22; // rdx
  _BYTE *v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rax
  __int64 v26; // r12
  __int64 v27; // rdx
  _BYTE *v28; // rax
  unsigned int v29; // r12d
  __int64 v30; // r12
  __int64 v31; // rdx
  _BYTE *v32; // rax
  unsigned int v33; // r12d
  bool v35; // r12
  unsigned int v36; // r15d
  __int64 v37; // rax
  unsigned int v38; // r12d
  bool v39; // r12
  unsigned int v40; // r15d
  __int64 v41; // rax
  unsigned int v42; // r12d
  int v43; // [rsp-3Ch] [rbp-3Ch]
  int v44; // [rsp-3Ch] [rbp-3Ch]

  if ( !a2 )
    return 0;
  v2 = *(_BYTE **)(a2 - 64);
  if ( *v2 != 58 )
    goto LABEL_3;
  v5 = (_BYTE *)*((_QWORD *)v2 - 8);
  if ( *v5 != 56 )
    goto LABEL_3;
  v6 = *((_QWORD *)v5 - 8);
  if ( !v6 )
    goto LABEL_3;
  **(_QWORD **)a1 = v6;
  v7 = *((_QWORD *)v5 - 4);
  if ( !v7 )
    BUG();
  if ( *(_BYTE *)v7 != 17 )
  {
    v22 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
    if ( (unsigned int)v22 > 1 )
      goto LABEL_3;
    if ( *(_BYTE *)v7 > 0x15u )
      goto LABEL_3;
    v23 = sub_AD7630(v7, 1, v22);
    v7 = (__int64)v23;
    if ( !v23 || *v23 != 17 )
      goto LABEL_3;
  }
  v8 = *(_DWORD *)(v7 + 32);
  if ( v8 > 0x40 )
  {
    if ( v8 - (unsigned int)sub_C444A0(v7 + 24) > 0x40 )
      goto LABEL_3;
    v9 = **(_QWORD **)(v7 + 24);
  }
  else
  {
    v9 = *(_QWORD *)(v7 + 24);
  }
  if ( *(_QWORD *)(a1 + 8) != v9 )
    goto LABEL_3;
  v10 = *((_QWORD *)v2 - 4);
  if ( *(_BYTE *)v10 == 17 )
  {
    v11 = *(_DWORD *)(v10 + 32);
    v12 = v11 <= 0x40 ? *(_QWORD *)(v10 + 24) == 1 : v11 - 1 == (unsigned int)sub_C444A0(v10 + 24);
  }
  else
  {
    v26 = *(_QWORD *)(v10 + 8);
    v27 = (unsigned int)*(unsigned __int8 *)(v26 + 8) - 17;
    if ( (unsigned int)v27 > 1 || *(_BYTE *)v10 > 0x15u )
      goto LABEL_3;
    v28 = sub_AD7630(v10, 0, v27);
    if ( !v28 || *v28 != 17 )
    {
      if ( *(_BYTE *)(v26 + 8) == 17 )
      {
        v43 = *(_DWORD *)(v26 + 32);
        if ( v43 )
        {
          v35 = 0;
          v36 = 0;
          while ( 1 )
          {
            v37 = sub_AD69F0((unsigned __int8 *)v10, v36);
            if ( !v37 )
              break;
            if ( *(_BYTE *)v37 != 13 )
            {
              if ( *(_BYTE *)v37 != 17 )
                break;
              v38 = *(_DWORD *)(v37 + 32);
              v35 = v38 <= 0x40 ? *(_QWORD *)(v37 + 24) == 1 : v38 - 1 == (unsigned int)sub_C444A0(v37 + 24);
              if ( !v35 )
                break;
            }
            if ( v43 == ++v36 )
            {
              if ( v35 )
                goto LABEL_18;
              goto LABEL_3;
            }
          }
        }
      }
      goto LABEL_3;
    }
    v29 = *((_DWORD *)v28 + 8);
    v12 = v29 <= 0x40 ? *((_QWORD *)v28 + 3) == 1 : v29 - 1 == (unsigned int)sub_C444A0((__int64)(v28 + 24));
  }
  if ( !v12 )
  {
LABEL_3:
    v3 = *(_BYTE **)(a2 - 32);
    goto LABEL_4;
  }
LABEL_18:
  v13 = *(__int64 **)(a1 + 16);
  if ( v13 )
    *v13 = v10;
  v3 = *(_BYTE **)(a2 - 32);
  if ( **(_BYTE ***)(a1 + 24) == v3 )
    return 1;
LABEL_4:
  if ( *v3 != 58 )
    return 0;
  v14 = (_BYTE *)*((_QWORD *)v3 - 8);
  if ( *v14 != 56 )
    return 0;
  v15 = *((_QWORD *)v14 - 8);
  if ( !v15 )
    return 0;
  **(_QWORD **)a1 = v15;
  v16 = *((_QWORD *)v14 - 4);
  if ( !v16 )
    BUG();
  if ( *(_BYTE *)v16 != 17 )
  {
    v24 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v16 + 8) + 8LL) - 17;
    if ( (unsigned int)v24 > 1 )
      return 0;
    if ( *(_BYTE *)v16 > 0x15u )
      return 0;
    v25 = sub_AD7630(v16, 1, v24);
    v16 = (__int64)v25;
    if ( !v25 || *v25 != 17 )
      return 0;
  }
  v17 = *(_DWORD *)(v16 + 32);
  if ( v17 > 0x40 )
  {
    if ( v17 - (unsigned int)sub_C444A0(v16 + 24) > 0x40 )
      return 0;
    v18 = **(_QWORD **)(v16 + 24);
  }
  else
  {
    v18 = *(_QWORD *)(v16 + 24);
  }
  if ( *(_QWORD *)(a1 + 8) != v18 )
    return 0;
  v19 = *((_QWORD *)v3 - 4);
  if ( *(_BYTE *)v19 != 17 )
  {
    v30 = *(_QWORD *)(v19 + 8);
    v31 = (unsigned int)*(unsigned __int8 *)(v30 + 8) - 17;
    if ( (unsigned int)v31 <= 1 && *(_BYTE *)v19 <= 0x15u )
    {
      v32 = sub_AD7630(v19, 0, v31);
      if ( !v32 || *v32 != 17 )
      {
        if ( *(_BYTE *)(v30 + 8) == 17 )
        {
          v44 = *(_DWORD *)(v30 + 32);
          if ( v44 )
          {
            v39 = 0;
            v40 = 0;
            while ( 1 )
            {
              v41 = sub_AD69F0((unsigned __int8 *)v19, v40);
              if ( !v41 )
                break;
              if ( *(_BYTE *)v41 != 13 )
              {
                if ( *(_BYTE *)v41 != 17 )
                  break;
                v42 = *(_DWORD *)(v41 + 32);
                v39 = v42 <= 0x40 ? *(_QWORD *)(v41 + 24) == 1 : v42 - 1 == (unsigned int)sub_C444A0(v41 + 24);
                if ( !v39 )
                  break;
              }
              if ( v44 == ++v40 )
              {
                if ( !v39 )
                  return 0;
                goto LABEL_32;
              }
            }
          }
        }
        return 0;
      }
      v33 = *((_DWORD *)v32 + 8);
      if ( v33 <= 0x40 ? *((_QWORD *)v32 + 3) == 1 : v33 - 1 == (unsigned int)sub_C444A0((__int64)(v32 + 24)) )
        goto LABEL_32;
    }
    return 0;
  }
  v20 = *(_DWORD *)(v19 + 32);
  if ( v20 <= 0x40 )
  {
    if ( *(_QWORD *)(v19 + 24) == 1 )
      goto LABEL_32;
    return 0;
  }
  if ( (unsigned int)sub_C444A0(v19 + 24) != v20 - 1 )
    return 0;
LABEL_32:
  v21 = *(__int64 **)(a1 + 16);
  if ( v21 )
    *v21 = v19;
  return **(_QWORD **)(a1 + 24) == *(_QWORD *)(a2 - 64);
}

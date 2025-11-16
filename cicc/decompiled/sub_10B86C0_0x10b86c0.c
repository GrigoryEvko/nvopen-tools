// Function: sub_10B86C0
// Address: 0x10b86c0
//
__int64 __fastcall sub_10B86C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int16 v9; // r13
  __int64 v10; // rdx
  bool v11; // al
  unsigned int v12; // eax
  __int64 v13; // rax
  int v14; // eax
  int v15; // edx
  void **v16; // rax
  unsigned __int8 *v17; // rdx
  __int64 v18; // r8
  void **v19; // rsi
  void **v20; // rax
  __int64 v21; // r8
  char v22; // dl
  _BYTE *v23; // rsi
  unsigned int v24; // r8d
  void **v25; // rax
  void **v26; // rsi
  char v27; // al
  unsigned int v28; // r8d
  void *v29; // rax
  _BYTE *v30; // rsi
  unsigned int v31; // r8d
  void **v32; // rax
  void **v33; // rsi
  char v34; // al
  unsigned int v35; // r8d
  void *v36; // rax
  _BYTE *v37; // rsi
  char v38; // [rsp-9Dh] [rbp-9Dh]
  char v39; // [rsp-9Dh] [rbp-9Dh]
  int v40; // [rsp-9Ch] [rbp-9Ch]
  int v41; // [rsp-9Ch] [rbp-9Ch]
  unsigned __int8 *v42; // [rsp-90h] [rbp-90h]
  char v43; // [rsp-90h] [rbp-90h]
  char v44; // [rsp-90h] [rbp-90h]
  unsigned int v45; // [rsp-90h] [rbp-90h]
  unsigned int v46; // [rsp-90h] [rbp-90h]
  unsigned __int8 *v47; // [rsp-90h] [rbp-90h]
  __int64 v48; // [rsp-88h] [rbp-88h]
  __int64 v49; // [rsp-88h] [rbp-88h]
  void **v50; // [rsp-88h] [rbp-88h]
  __int64 v51; // [rsp-88h] [rbp-88h]
  void **v52; // [rsp-88h] [rbp-88h]
  unsigned int v53; // [rsp-88h] [rbp-88h]
  char v54; // [rsp-88h] [rbp-88h]
  unsigned __int8 *v55; // [rsp-88h] [rbp-88h]
  unsigned int v56; // [rsp-88h] [rbp-88h]
  __int64 v57; // [rsp-80h] [rbp-80h]
  __int64 v58; // [rsp-70h] [rbp-70h]
  _WORD v59[52]; // [rsp-68h] [rbp-68h] BYREF

  if ( (*(_WORD *)(a2 + 2) & 0x3F) != 7 )
    return 0;
  v6 = *(_QWORD *)(a3 - 64);
  v7 = *(_QWORD *)(a2 - 32);
  v8 = *(_QWORD *)(a3 - 32);
  v9 = *(_WORD *)(a3 + 2);
  v57 = v6;
  if ( *(_BYTE *)v7 == 18 )
  {
    v48 = *(_QWORD *)(a2 - 32);
    if ( *(void **)(v7 + 24) == sub_C33340() )
      v10 = *(_QWORD *)(v48 + 32);
    else
      v10 = v48 + 24;
    v11 = (*(_BYTE *)(v10 + 20) & 7) == 3;
  }
  else
  {
    v49 = *(_QWORD *)(v7 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v49 + 8) - 17 > 1 || *(_BYTE *)v7 > 0x15u )
      return 0;
    v42 = *(unsigned __int8 **)(a2 - 32);
    v16 = (void **)sub_AD7630(v7, 0, v7);
    v17 = v42;
    v18 = v49;
    if ( !v16 || (v50 = v16, *(_BYTE *)v16 != 18) )
    {
      if ( *(_BYTE *)(v18 + 8) == 17 )
      {
        v41 = *(_DWORD *)(v18 + 32);
        if ( v41 )
        {
          v39 = 0;
          v31 = 0;
          while ( 1 )
          {
            v46 = v31;
            v55 = v17;
            v32 = (void **)sub_AD69F0(v17, v31);
            v33 = v32;
            if ( !v32 )
              break;
            v34 = *(_BYTE *)v32;
            v17 = v55;
            v35 = v46;
            if ( v34 != 13 )
            {
              v47 = v55;
              v56 = v35;
              if ( v34 != 18 )
                return 0;
              v36 = sub_C33340();
              v35 = v56;
              v17 = v47;
              v37 = v33[3] == v36 ? v33[4] : v33 + 3;
              if ( (v37[20] & 7) != 3 )
                return 0;
              v39 = 1;
            }
            v31 = v35 + 1;
            if ( v41 == v31 )
            {
              if ( v39 )
                goto LABEL_8;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    if ( v16[3] == sub_C33340() )
      v19 = (void **)v50[4];
    else
      v19 = v50 + 3;
    v11 = (*((_BYTE *)v19 + 20) & 7) == 3;
  }
  if ( !v11 )
    return 0;
LABEL_8:
  LOBYTE(v12) = sub_B535C0(v9 & 0x3F);
  if ( !(_BYTE)v12 )
    return 0;
  if ( *(_BYTE *)v8 != 18 )
  {
    v43 = v12;
    v51 = *(_QWORD *)(v8 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v51 + 8) - 17 <= 1 && *(_BYTE *)v8 <= 0x15u )
    {
      v20 = (void **)sub_AD7630(v8, 0, v12);
      v21 = v51;
      v22 = v43;
      if ( !v20 || (v52 = v20, *(_BYTE *)v20 != 18) )
      {
        if ( *(_BYTE *)(v21 + 8) == 17 )
        {
          v40 = *(_DWORD *)(v21 + 32);
          if ( v40 )
          {
            v38 = 0;
            v24 = 0;
            while ( 1 )
            {
              v44 = v22;
              v53 = v24;
              v25 = (void **)sub_AD69F0((unsigned __int8 *)v8, v24);
              v26 = v25;
              if ( !v25 )
                break;
              v27 = *(_BYTE *)v25;
              v28 = v53;
              v22 = v44;
              if ( v27 != 13 )
              {
                v45 = v53;
                v54 = v22;
                if ( v27 != 18 )
                  return 0;
                v29 = sub_C33340();
                v22 = v54;
                v28 = v45;
                v30 = v26[3] == v29 ? v26[4] : v26 + 3;
                if ( (v30[20] & 7) != 0 )
                  return 0;
                v38 = v54;
              }
              v24 = v28 + 1;
              if ( v40 == v24 )
              {
                if ( v38 )
                  goto LABEL_14;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v23 = v20[3] == sub_C33340() ? v52[4] : v52 + 3;
      if ( (v23[20] & 7) == 0 )
        goto LABEL_14;
    }
    return 0;
  }
  if ( *(void **)(v8 + 24) == sub_C33340() )
    v13 = *(_QWORD *)(v8 + 32);
  else
    v13 = v8 + 24;
  if ( (*(_BYTE *)(v13 + 20) & 7) != 0 )
    return 0;
LABEL_14:
  v59[16] = 257;
  BYTE4(v58) = 1;
  v14 = *(_BYTE *)(a3 + 1) >> 1;
  if ( v14 == 127 )
    v14 = -1;
  v15 = *(_BYTE *)(a2 + 1) >> 1;
  if ( v15 != 127 )
    v14 &= v15;
  LODWORD(v58) = v14;
  return sub_B35C90(a1, v9 & 7, v57, v8, (__int64)v59, 0, v58, 0);
}

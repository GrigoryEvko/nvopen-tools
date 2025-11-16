// Function: sub_1464C80
// Address: 0x1464c80
//
void __fastcall sub_1464C80(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  __int64 *v4; // rdi
  __int64 *v5; // r8
  unsigned int v6; // eax
  __int64 v7; // r12
  char v8; // dl
  int v9; // r15d
  char v10; // r10
  unsigned int v11; // r13d
  int v12; // edx
  unsigned int v13; // edi
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rsi
  int v17; // eax
  int v18; // edx
  __int64 v19; // rsi
  unsigned int v20; // r13d
  __int64 *v21; // rax
  __int64 v22; // rcx
  __int64 *v23; // rsi
  __int64 *v24; // rax
  __int64 *v25; // rcx
  int v26; // eax
  int v27; // edi
  __int64 v28; // r9
  int v29; // r10d
  __int64 v30; // rax
  __int64 v31; // [rsp-1A0h] [rbp-1A0h]
  char v32; // [rsp-1A0h] [rbp-1A0h]
  char v33; // [rsp-1A0h] [rbp-1A0h]
  void *v34; // [rsp-198h] [rbp-198h] BYREF
  __int64 v35; // [rsp-190h] [rbp-190h] BYREF
  __int64 v36; // [rsp-180h] [rbp-180h]
  void *v37; // [rsp-168h] [rbp-168h] BYREF
  __int64 v38; // [rsp-160h] [rbp-160h] BYREF
  __int64 v39; // [rsp-150h] [rbp-150h]
  __int64 v40; // [rsp-138h] [rbp-138h] BYREF
  __int64 *v41; // [rsp-130h] [rbp-130h]
  __int64 *v42; // [rsp-128h] [rbp-128h]
  __int64 v43; // [rsp-120h] [rbp-120h]
  int v44; // [rsp-118h] [rbp-118h]
  _BYTE v45[72]; // [rsp-110h] [rbp-110h] BYREF
  _QWORD *v46; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v47; // [rsp-C0h] [rbp-C0h]
  _QWORD v48[23]; // [rsp-B8h] [rbp-B8h] BYREF

  if ( *(_BYTE *)(a2 + 16) <= 0x17u )
    return;
  v2 = v48;
  v4 = (__int64 *)v45;
  v5 = (__int64 *)v45;
  v46 = v48;
  v48[0] = a2;
  v40 = 0;
  v41 = (__int64 *)v45;
  v42 = (__int64 *)v45;
  v43 = 8;
  v44 = 0;
  v47 = 0x1000000001LL;
  v6 = 1;
  while ( 1 )
  {
    v7 = v2[v6 - 1];
    LODWORD(v47) = v6 - 1;
    if ( v4 != v5 )
      goto LABEL_4;
    v23 = &v4[HIDWORD(v43)];
    if ( v23 == v4 )
    {
LABEL_35:
      if ( HIDWORD(v43) >= (unsigned int)v43 )
      {
LABEL_4:
        sub_16CCBA0(&v40, v7);
        v5 = v42;
        v4 = v41;
        if ( !v8 )
          goto LABEL_20;
      }
      else
      {
        ++HIDWORD(v43);
        *v23 = v7;
        ++v40;
      }
LABEL_5:
      v9 = *(_DWORD *)(a1 + 168);
      if ( v9 )
      {
        v31 = *(_QWORD *)(a1 + 152);
        sub_1457D90(&v34, -8, 0);
        sub_1457D90(&v37, -16, 0);
        v10 = 1;
        v11 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
        v12 = v9 - 1;
        v13 = (v9 - 1) & v11;
        v14 = 48LL * v13;
        v15 = v31 + v14;
        v16 = *(_QWORD *)(v31 + v14 + 24);
        if ( v7 != v16 )
        {
          v28 = v31 + v14;
          v29 = 1;
          v15 = 0;
          while ( v16 != v36 )
          {
            if ( v16 != v39 || v15 )
              v28 = v15;
            v13 = v12 & (v29 + v13);
            v15 = v31 + 48LL * v13;
            v16 = *(_QWORD *)(v15 + 24);
            if ( v7 == v16 )
            {
              v10 = 1;
              goto LABEL_7;
            }
            v30 = v28;
            ++v29;
            v28 = v31 + 48LL * v13;
            v15 = v30;
          }
          v10 = 0;
          if ( !v15 )
            v15 = v28;
        }
LABEL_7:
        v37 = &unk_49EE2B0;
        if ( v39 != -8 && v39 != 0 && v39 != -16 )
        {
          v32 = v10;
          sub_1649B30(&v38);
          v10 = v32;
        }
        v34 = &unk_49EE2B0;
        if ( v36 != 0 && v36 != -8 && v36 != -16 )
        {
          v33 = v10;
          sub_1649B30(&v35);
          v10 = v33;
        }
        if ( v10 )
        {
          if ( v15 != *(_QWORD *)(a1 + 152) + 48LL * *(unsigned int *)(a1 + 168) )
          {
            sub_1464220(a1, *(_QWORD *)(v15 + 24));
            sub_1459590(a1, *(_QWORD *)(v15 + 40));
            if ( *(_BYTE *)(v7 + 16) == 77 )
            {
              v17 = *(_DWORD *)(a1 + 616);
              if ( v17 )
              {
                v18 = v17 - 1;
                v19 = *(_QWORD *)(a1 + 600);
                v20 = (v17 - 1) & v11;
                v21 = (__int64 *)(v19 + 16LL * v20);
                v22 = *v21;
                if ( v7 == *v21 )
                {
LABEL_18:
                  *v21 = -16;
                  --*(_DWORD *)(a1 + 608);
                  ++*(_DWORD *)(a1 + 612);
                }
                else
                {
                  v26 = 1;
                  while ( v22 != -8 )
                  {
                    v27 = v26 + 1;
                    v20 = v18 & (v26 + v20);
                    v21 = (__int64 *)(v19 + 16LL * v20);
                    v22 = *v21;
                    if ( v7 == *v21 )
                      goto LABEL_18;
                    v26 = v27;
                  }
                }
              }
            }
          }
        }
      }
      sub_1453C10(v7, (__int64)&v46);
      v5 = v42;
      v4 = v41;
      goto LABEL_20;
    }
    v24 = v4;
    v25 = 0;
    while ( v7 != *v24 )
    {
      if ( *v24 == -2 )
        v25 = v24;
      if ( v23 == ++v24 )
      {
        if ( !v25 )
          goto LABEL_35;
        *v25 = v7;
        --v44;
        ++v40;
        goto LABEL_5;
      }
    }
LABEL_20:
    v6 = v47;
    if ( !(_DWORD)v47 )
      break;
    v2 = v46;
  }
  if ( v4 != v5 )
    _libc_free((unsigned __int64)v5);
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
}

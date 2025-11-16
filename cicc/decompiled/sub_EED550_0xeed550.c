// Function: sub_EED550
// Address: 0xeed550
//
_QWORD *__fastcall sub_EED550(__int64 a1)
{
  _BYTE *v1; // rax
  _BYTE *v2; // rdx
  char v4; // dl
  unsigned __int8 v5; // bl
  char v6; // r12
  char *v7; // rax
  char *v8; // r15
  char v9; // al
  const char *v10; // r13
  __int64 v11; // r13
  unsigned __int8 *v12; // r12
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD **v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // r9
  __int64 v28; // r9
  __int64 *v29; // rax
  __int64 v30; // rax
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // rsi
  char v34; // al
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 *v37; // rdx
  char v38; // [rsp-100h] [rbp-100h]
  _QWORD *v39; // [rsp-F8h] [rbp-F8h]
  unsigned __int8 *v40; // [rsp-F0h] [rbp-F0h]
  __int64 v41; // [rsp-E8h] [rbp-E8h]
  _QWORD *v42; // [rsp-E8h] [rbp-E8h]
  size_t v43; // [rsp-E0h] [rbp-E0h]
  _QWORD *v44; // [rsp-E0h] [rbp-E0h]
  __int64 v45; // [rsp-E0h] [rbp-E0h]
  _QWORD *v46; // [rsp-E0h] [rbp-E0h]
  __int64 *v47; // [rsp-D0h] [rbp-D0h] BYREF
  _QWORD *v48; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v49; // [rsp-C0h] [rbp-C0h]
  _QWORD v50[23]; // [rsp-B8h] [rbp-B8h] BYREF

  v1 = *(_BYTE **)a1;
  v2 = *(_BYTE **)(a1 + 8);
  if ( v2 == *(_BYTE **)a1 || *v1 != 102 )
    return 0;
  *(_QWORD *)a1 = v1 + 1;
  if ( v1 + 1 == v2 )
    return 0;
  v4 = v1[1];
  if ( v4 == 108 )
  {
    v5 = 1;
    v6 = 0;
LABEL_10:
    *(_QWORD *)a1 = v1 + 2;
    v7 = sub_EE33C0(a1);
    v8 = v7;
    if ( v7 )
    {
      v9 = v7[2];
      if ( v9 == 2 || v9 == 4 && (v10 = (const char *)*((_QWORD *)v8 + 1), v10[strlen(v10) - 1] == 42) )
      {
        v11 = sub_EEA9F0(a1);
        if ( v11 )
        {
          v41 = 0;
          v39 = 0;
          if ( !v6 )
            goto LABEL_16;
          v32 = sub_EEA9F0(a1);
          v39 = (_QWORD *)v32;
          if ( v32 )
          {
            v41 = v32;
            if ( v5 )
            {
              v33 = v11;
              v41 = v11;
              v11 = v32;
              v39 = (_QWORD *)v33;
            }
LABEL_16:
            v12 = (unsigned __int8 *)*((_QWORD *)v8 + 1);
            v40 = v12;
            v43 = strlen((const char *)v12);
            v16 = v43;
            if ( (unsigned __int8)v8[2] <= 0xAu )
            {
              v16 = v43 - 8;
              if ( v43 == 8 )
              {
                v34 = *(_BYTE *)(a1 + 937);
                v50[0] = 71;
                v38 = v34;
                v48 = v50;
                v49 = 0x2000000002LL;
                sub_D953B0((__int64)&v48, v5, v13, 0, v14, v15);
                v40 = v12 + 8;
                goto LABEL_36;
              }
              if ( v12[8] != 32 )
              {
                v38 = *(_BYTE *)(a1 + 937);
                v48 = v50;
                v49 = 0x2000000002LL;
                v50[0] = 71;
                sub_D953B0((__int64)&v48, v5, v13, v16, v14, v15);
                v40 = v12 + 8;
                v43 -= 8LL;
LABEL_20:
                sub_C653C0((__int64)&v48, v40, v43);
                goto LABEL_21;
              }
              v40 = v12 + 9;
              v43 -= 9LL;
            }
            v31 = *(_BYTE *)(a1 + 937);
            v50[0] = 71;
            v38 = v31;
            v48 = v50;
            v49 = 0x2000000002LL;
            sub_D953B0((__int64)&v48, v5, v13, v16, v14, v15);
            if ( v43 )
              goto LABEL_20;
LABEL_36:
            sub_C653C0((__int64)&v48, 0, 0);
            v43 = 0;
LABEL_21:
            sub_D953B0((__int64)&v48, v11, v17, v18, v19, v20);
            sub_D953B0((__int64)&v48, v41, v21, v22, v23, v24);
            v25 = &v48;
            v26 = sub_C65B40(a1 + 904, (__int64)&v48, (__int64 *)&v47, (__int64)off_497B2F0);
            v27 = v26;
            if ( v26 )
            {
              v28 = (__int64)(v26 + 1);
              if ( v48 != v50 )
              {
                v44 = v26 + 1;
                _libc_free(v48, &v48);
                v28 = (__int64)v44;
              }
              v48 = (_QWORD *)v28;
              v45 = v28;
              v29 = sub_EE6840(a1 + 944, (__int64 *)&v48);
              v27 = (_QWORD *)v45;
              if ( v29 )
              {
                v30 = v29[1];
                if ( v30 )
                  v27 = (_QWORD *)v30;
              }
              if ( *(_QWORD **)(a1 + 928) == v27 )
                *(_BYTE *)(a1 + 936) = 1;
            }
            else
            {
              if ( v38 )
              {
                v35 = sub_CD1D40((__int64 *)(a1 + 808), 64, 3);
                *(_QWORD *)v35 = 0;
                v25 = (_QWORD **)v35;
                v36 = v35 + 8;
                *(_WORD *)(v35 + 16) = 16455;
                LOBYTE(v35) = *(_BYTE *)(v35 + 18);
                v25[3] = (_QWORD *)v11;
                v37 = v47;
                *((_BYTE *)v25 + 56) = v5;
                v42 = (_QWORD *)v36;
                *((_BYTE *)v25 + 18) = v35 & 0xF0 | 5;
                v25[1] = &unk_49E0928;
                v25[4] = v39;
                v25[5] = (_QWORD *)v43;
                v25[6] = v40;
                sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v25, v37, (__int64)off_497B2F0);
                v27 = v42;
              }
              if ( v48 != v50 )
              {
                v46 = v27;
                _libc_free(v48, v25);
                v27 = v46;
              }
              *(_QWORD *)(a1 + 920) = v27;
            }
            return v27;
          }
        }
      }
    }
    return 0;
  }
  if ( v4 > 108 )
  {
    v27 = 0;
    if ( v4 != 114 )
      return v27;
    v5 = 0;
    v6 = 0;
    goto LABEL_10;
  }
  if ( v4 == 76 )
  {
    v5 = 1;
    v6 = 1;
    goto LABEL_10;
  }
  if ( v4 == 82 )
  {
    v5 = 0;
    v6 = 1;
    goto LABEL_10;
  }
  return 0;
}

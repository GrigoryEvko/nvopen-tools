// Function: sub_2E8F690
// Address: 0x2e8f690
//
__int64 __fastcall sub_2E8F690(__int64 a1, unsigned int a2, _QWORD *a3, char a4)
{
  bool v4; // r10
  __int32 v5; // r14d
  int v7; // r8d
  int v8; // r8d
  __int64 v9; // rbx
  int v10; // r14d
  int v11; // r8d
  __int64 v13; // rax
  unsigned __int8 v14; // dl
  unsigned int v15; // ecx
  __int64 v16; // rax
  unsigned int v17; // r15d
  __int64 v18; // rax
  bool v19; // zf
  __int64 v20; // r13
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // r9
  __int16 *v25; // rax
  int v26; // esi
  __int64 v27; // rax
  __int16 v28; // di
  __int32 v29; // esi
  unsigned int v30; // eax
  __int16 *v31; // rax
  int v32; // r9d
  __int64 v33; // rax
  bool v34; // al
  __int64 v35; // r9
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned __int8 v38; // [rsp+Eh] [rbp-D2h]
  bool v39; // [rsp+Fh] [rbp-D1h]
  __int64 v40; // [rsp+10h] [rbp-D0h]
  __int64 v41; // [rsp+18h] [rbp-C8h]
  __int64 v42; // [rsp+30h] [rbp-B0h]
  unsigned int v44; // [rsp+40h] [rbp-A0h]
  unsigned int v46; // [rsp+5Ch] [rbp-84h] BYREF
  _BYTE *v47; // [rsp+60h] [rbp-80h] BYREF
  __int64 v48; // [rsp+68h] [rbp-78h]
  _BYTE v49[16]; // [rsp+70h] [rbp-70h] BYREF
  __m128i v50; // [rsp+80h] [rbp-60h] BYREF
  __int64 v51; // [rsp+90h] [rbp-50h]
  __int64 v52; // [rsp+98h] [rbp-48h]
  __int64 v53; // [rsp+A0h] [rbp-40h]
  __int16 v54; // [rsp+A8h] [rbp-38h]

  v4 = 0;
  v5 = a2;
  if ( a2 - 1 <= 0x3FFFFFFE )
  {
    sub_E922F0(a3, a2);
    v4 = 2 * v22 != 2;
  }
  v7 = *(_DWORD *)(a1 + 40);
  v47 = v49;
  v48 = 0x400000000LL;
  v8 = v7 & 0xFFFFFF;
  if ( v8 )
  {
    v9 = 0;
    v42 = 24LL * a2;
    v10 = v8;
    v11 = 0;
    while ( 1 )
    {
      v13 = *(_QWORD *)(a1 + 32) + 40 * v9;
      if ( *(_BYTE *)v13 )
        goto LABEL_8;
      v14 = *(_BYTE *)(v13 + 3);
      if ( (v14 & 0x10) == 0 )
        goto LABEL_8;
      v15 = *(_DWORD *)(v13 + 8);
      if ( !v15 )
        goto LABEL_8;
      if ( a2 == v15 )
      {
        ++v9;
        v11 = 1;
        *(_BYTE *)(v13 + 3) = v14 | 0x40;
        if ( v10 == (_DWORD)v9 )
        {
LABEL_14:
          v16 = (unsigned int)v48;
          v5 = a2;
          v17 = v11;
          if ( !(_DWORD)v48 )
          {
LABEL_22:
            if ( a4 != 1 || (_BYTE)v17 )
              goto LABEL_25;
            goto LABEL_24;
          }
          while ( 1 )
          {
LABEL_19:
            v20 = *(unsigned int *)&v47[4 * v16 - 4];
            v18 = 40 * v20 + *(_QWORD *)(a1 + 32);
            if ( (*(_BYTE *)(v18 + 3) & 0x20) == 0 )
              goto LABEL_18;
            if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 <= 1 && (int)sub_2E890A0(a1, v20, 0) >= 0 )
              break;
            sub_2E8A650(a1, v20);
            v19 = (_DWORD)v48 == 1;
            v16 = (unsigned int)(v48 - 1);
            LODWORD(v48) = v48 - 1;
            if ( v19 )
              goto LABEL_22;
          }
          v18 = 40 * v20 + *(_QWORD *)(a1 + 32);
LABEL_18:
          *(_BYTE *)(v18 + 3) &= ~0x40u;
          v19 = (_DWORD)v48 == 1;
          v16 = (unsigned int)(v48 - 1);
          LODWORD(v48) = v48 - 1;
          if ( v19 )
            goto LABEL_22;
          goto LABEL_19;
        }
      }
      else
      {
        if ( v4 && (((v14 & 0x10) != 0) & (v14 >> 6)) != 0 && v15 - 1 <= 0x3FFFFFFE )
        {
          v38 = v11;
          v39 = v4;
          v23 = a3[7];
          v24 = a3[1];
          v46 = *(_DWORD *)(v13 + 8);
          v44 = v15;
          v40 = v24;
          v41 = v23;
          v25 = (__int16 *)(v23 + 2LL * *(unsigned int *)(v24 + v42 + 8));
          v26 = *v25;
          v27 = (__int64)(v25 + 1);
          LODWORD(v52) = 0;
          v53 = 0;
          v28 = v26;
          v29 = a2 + v26;
          v50.m128i_i32[0] = v29;
          if ( !v28 )
            v27 = 0;
          LOWORD(v51) = v29;
          v50.m128i_i64[1] = v27;
          v54 = 0;
          LOBYTE(v30) = sub_2E46590(v50.m128i_i32, (int *)&v46);
          if ( (_BYTE)v30 )
          {
            v17 = v30;
            goto LABEL_25;
          }
          v46 = a2;
          v31 = (__int16 *)(v41 + 2LL * *(unsigned int *)(v40 + 24LL * v44 + 8));
          v32 = *v31;
          v33 = (__int64)(v31 + 1);
          if ( !(_WORD)v32 )
            v33 = 0;
          v50.m128i_i32[0] = v32 + v44;
          LOWORD(v51) = v32 + v44;
          v50.m128i_i64[1] = v33;
          v34 = sub_2E46590(v50.m128i_i32, (int *)&v46);
          v4 = v39;
          v11 = v38;
          if ( v34 )
          {
            v36 = (unsigned int)v48;
            v37 = (unsigned int)v48 + 1LL;
            if ( v37 > HIDWORD(v48) )
            {
              sub_C8D5F0((__int64)&v47, v49, v37, 4u, v38, v35);
              v36 = (unsigned int)v48;
              v11 = v38;
              v4 = v39;
            }
            *(_DWORD *)&v47[4 * v36] = v9;
            LODWORD(v48) = v48 + 1;
          }
        }
LABEL_8:
        if ( v10 == (_DWORD)++v9 )
          goto LABEL_14;
      }
    }
  }
  if ( a4 )
  {
LABEL_24:
    v50.m128i_i32[2] = v5;
    v17 = 1;
    v50.m128i_i64[0] = 1879048192;
    v51 = 0;
    v52 = 0;
    v53 = 0;
    sub_2E8F270(a1, &v50);
LABEL_25:
    if ( v47 != v49 )
      _libc_free((unsigned __int64)v47);
  }
  else
  {
    return 0;
  }
  return v17;
}

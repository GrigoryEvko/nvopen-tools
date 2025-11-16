// Function: sub_255D150
// Address: 0x255d150
//
__int64 __fastcall sub_255D150(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  unsigned int v6; // eax
  unsigned __int64 v7; // r8
  __m128i *v8; // rax
  __m128i *v9; // rdx
  unsigned __int64 v10; // rax
  int *v11; // rdx
  char v12; // [rsp+Bh] [rbp-175h] BYREF
  int v13; // [rsp+Ch] [rbp-174h] BYREF
  _QWORD v14[4]; // [rsp+10h] [rbp-170h] BYREF
  _QWORD v15[2]; // [rsp+30h] [rbp-150h] BYREF
  __int128 v16; // [rsp+40h] [rbp-140h] BYREF
  __int128 v17; // [rsp+50h] [rbp-130h] BYREF
  __int128 *v18; // [rsp+60h] [rbp-120h]
  __int128 *v19; // [rsp+68h] [rbp-118h]
  __int64 v20; // [rsp+70h] [rbp-110h]
  void *v21; // [rsp+78h] [rbp-108h]
  __int64 v22; // [rsp+80h] [rbp-100h]
  void *v23; // [rsp+90h] [rbp-F0h]
  void *v24; // [rsp+98h] [rbp-E8h]
  int v25; // [rsp+A0h] [rbp-E0h]
  unsigned int v26; // [rsp+A4h] [rbp-DCh]
  char v27[8]; // [rsp+A8h] [rbp-D8h] BYREF
  int v28; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned __int64 v29; // [rsp+B8h] [rbp-C8h]
  __m128i *v30; // [rsp+C0h] [rbp-C0h]
  int *v31; // [rsp+C8h] [rbp-B8h]
  __int64 v32; // [rsp+D0h] [rbp-B0h]
  _QWORD v33[18]; // [rsp+F0h] [rbp-90h] BYREF

  v20 = 0;
  v22 = 256;
  v16 = 0;
  v15[0] = &unk_4A16DD8;
  v15[1] = &unk_4A16D78;
  v18 = &v17;
  v19 = &v17;
  DWORD1(v16) = -1;
  v21 = &unk_4A16CD8;
  memset(v33, 0, 0x60u);
  v17 = 0;
  v13 = sub_250CB50((__int64 *)(a1 + 72), 0);
  v14[0] = &v13;
  v14[1] = a2;
  v14[2] = a1;
  v14[3] = v33;
  v12 = 0;
  if ( !(unsigned __int8)sub_2523890(
                           a2,
                           (__int64 (__fastcall *)(__int64, __int64 *))sub_258E7C0,
                           (__int64)v14,
                           a1,
                           1u,
                           &v12) )
  {
    DWORD1(v16) = v16;
    BYTE1(v22) = v22;
    if ( !LOBYTE(v33[11]) )
      goto LABEL_3;
    goto LABEL_20;
  }
  if ( LOBYTE(v33[11]) )
  {
    v6 = DWORD1(v16);
    if ( HIDWORD(v33[2]) <= DWORD1(v16) )
      v6 = HIDWORD(v33[2]);
    if ( v6 < (unsigned int)v16 )
      v6 = v16;
    DWORD1(v16) = v6;
    if ( !BYTE1(v33[10]) )
      BYTE1(v22) = v22;
    v7 = *((_QWORD *)&v17 + 1);
    v26 = v6;
    v25 = v16;
    v23 = &unk_4A16DD8;
    v24 = &unk_4A16D78;
    v28 = 0;
    v29 = 0;
    v30 = (__m128i *)&v28;
    v31 = &v28;
    v32 = 0;
    if ( *((_QWORD *)&v17 + 1) )
    {
      v8 = sub_25394D0(*((const __m128i **)&v17 + 1), (__int64)&v28);
      v7 = (unsigned __int64)v8;
      do
      {
        v9 = v8;
        v8 = (__m128i *)v8[1].m128i_i64[0];
      }
      while ( v8 );
      v30 = v9;
      v10 = v7;
      do
      {
        v11 = (int *)v10;
        v10 = *(_QWORD *)(v10 + 24);
      }
      while ( v10 );
      v31 = v11;
      v29 = v7;
      v32 = v20;
    }
    v23 = &unk_4A16DD8;
    sub_255C230((__int64)v27, v7);
    if ( LOBYTE(v33[11]) )
    {
LABEL_20:
      LOBYTE(v33[11]) = 0;
      v33[0] = &unk_4A16DD8;
      sub_255C230((__int64)&v33[3], v33[5]);
    }
  }
LABEL_3:
  v2 = sub_25538A0(a1 + 88, (__int64)v15);
  v3 = *((_QWORD *)&v17 + 1);
  v15[0] = &unk_4A16DD8;
  while ( v3 )
  {
    sub_255C230((__int64)&v16 + 8, *(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4);
  }
  return v2;
}

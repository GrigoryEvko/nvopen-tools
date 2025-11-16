// Function: sub_255D430
// Address: 0x255d430
//
__int64 __fastcall sub_255D430(__int64 a1, __int64 a2)
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
  __int64 v12; // [rsp+8h] [rbp-168h] BYREF
  _QWORD v13[4]; // [rsp+10h] [rbp-160h] BYREF
  _QWORD v14[2]; // [rsp+30h] [rbp-140h] BYREF
  __int128 v15; // [rsp+40h] [rbp-130h] BYREF
  __int128 v16; // [rsp+50h] [rbp-120h] BYREF
  __int128 *v17; // [rsp+60h] [rbp-110h]
  __int128 *v18; // [rsp+68h] [rbp-108h]
  __int64 v19; // [rsp+70h] [rbp-100h]
  void *v20; // [rsp+78h] [rbp-F8h]
  __int64 v21; // [rsp+80h] [rbp-F0h]
  void *v22; // [rsp+90h] [rbp-E0h]
  void *v23; // [rsp+98h] [rbp-D8h]
  int v24; // [rsp+A0h] [rbp-D0h]
  unsigned int v25; // [rsp+A4h] [rbp-CCh]
  char v26[8]; // [rsp+A8h] [rbp-C8h] BYREF
  int v27; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned __int64 v28; // [rsp+B8h] [rbp-B8h]
  __m128i *v29; // [rsp+C0h] [rbp-B0h]
  int *v30; // [rsp+C8h] [rbp-A8h]
  __int64 v31; // [rsp+D0h] [rbp-A0h]
  _QWORD v32[16]; // [rsp+F0h] [rbp-80h] BYREF

  v19 = 0;
  v13[1] = a2;
  v21 = 256;
  v14[0] = &unk_4A16DD8;
  v14[1] = &unk_4A16D78;
  v17 = &v16;
  v18 = &v16;
  v15 = 0;
  v20 = &unk_4A16CD8;
  memset(v32, 0, 0x60u);
  v13[0] = &v12;
  DWORD1(v15) = -1;
  v12 = 0;
  v13[2] = a1;
  v13[3] = v32;
  v16 = 0;
  if ( !(unsigned __int8)sub_2527330(
                           a2,
                           (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_258EBC0,
                           (__int64)v13,
                           a1,
                           1u,
                           1u) )
  {
    DWORD1(v15) = v15;
    BYTE1(v21) = v21;
    if ( !LOBYTE(v32[11]) )
      goto LABEL_3;
    goto LABEL_20;
  }
  if ( LOBYTE(v32[11]) )
  {
    v6 = DWORD1(v15);
    if ( HIDWORD(v32[2]) <= DWORD1(v15) )
      v6 = HIDWORD(v32[2]);
    if ( v6 < (unsigned int)v15 )
      v6 = v15;
    DWORD1(v15) = v6;
    if ( !BYTE1(v32[10]) )
      BYTE1(v21) = v21;
    v7 = *((_QWORD *)&v16 + 1);
    v25 = v6;
    v24 = v15;
    v22 = &unk_4A16DD8;
    v23 = &unk_4A16D78;
    v27 = 0;
    v28 = 0;
    v29 = (__m128i *)&v27;
    v30 = &v27;
    v31 = 0;
    if ( *((_QWORD *)&v16 + 1) )
    {
      v8 = sub_25394D0(*((const __m128i **)&v16 + 1), (__int64)&v27);
      v7 = (unsigned __int64)v8;
      do
      {
        v9 = v8;
        v8 = (__m128i *)v8[1].m128i_i64[0];
      }
      while ( v8 );
      v29 = v9;
      v10 = v7;
      do
      {
        v11 = (int *)v10;
        v10 = *(_QWORD *)(v10 + 24);
      }
      while ( v10 );
      v30 = v11;
      v28 = v7;
      v31 = v19;
    }
    v22 = &unk_4A16DD8;
    sub_255C230((__int64)v26, v7);
    if ( LOBYTE(v32[11]) )
    {
LABEL_20:
      LOBYTE(v32[11]) = 0;
      v32[0] = &unk_4A16DD8;
      sub_255C230((__int64)&v32[3], v32[5]);
    }
  }
LABEL_3:
  v2 = sub_25538A0(a1 + 88, (__int64)v14);
  v3 = *((_QWORD *)&v16 + 1);
  v14[0] = &unk_4A16DD8;
  while ( v3 )
  {
    sub_255C230((__int64)&v15 + 8, *(_QWORD *)(v3 + 24));
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 16);
    j_j___libc_free_0(v4);
  }
  return v2;
}

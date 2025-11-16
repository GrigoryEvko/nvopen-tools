// Function: sub_289F6B0
// Address: 0x289f6b0
//
void __fastcall sub_289F6B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int16 a5,
        __int64 a6,
        unsigned __int8 a7,
        char a8)
{
  __int64 v9; // rdx
  char v10; // r8
  __int64 v11; // r10
  __m128i v12; // xmm0
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned int *v19[2]; // [rsp+40h] [rbp-2D0h] BYREF
  _BYTE v20[32]; // [rsp+50h] [rbp-2C0h] BYREF
  __int64 v21; // [rsp+70h] [rbp-2A0h]
  __int64 v22; // [rsp+78h] [rbp-298h]
  __int16 v23; // [rsp+80h] [rbp-290h]
  __int64 v24; // [rsp+88h] [rbp-288h]
  void **v25; // [rsp+90h] [rbp-280h]
  void **v26; // [rsp+98h] [rbp-278h]
  __int64 v27; // [rsp+A0h] [rbp-270h]
  int v28; // [rsp+A8h] [rbp-268h]
  __int16 v29; // [rsp+ACh] [rbp-264h]
  char v30; // [rsp+AEh] [rbp-262h]
  __int64 v31; // [rsp+B0h] [rbp-260h]
  __int64 v32; // [rsp+B8h] [rbp-258h]
  void *v33; // [rsp+C0h] [rbp-250h] BYREF
  void *v34; // [rsp+C8h] [rbp-248h] BYREF
  char *v35; // [rsp+D0h] [rbp-240h] BYREF
  int v36; // [rsp+D8h] [rbp-238h]
  char v37; // [rsp+E0h] [rbp-230h] BYREF
  __m128i v38; // [rsp+160h] [rbp-1B0h] BYREF
  char v39; // [rsp+170h] [rbp-1A0h]
  unsigned __int64 v40[2]; // [rsp+180h] [rbp-190h] BYREF
  _BYTE v41[128]; // [rsp+190h] [rbp-180h] BYREF
  __m128i v42; // [rsp+210h] [rbp-100h]
  char v43; // [rsp+220h] [rbp-F0h]
  __m128i v44; // [rsp+230h] [rbp-E0h] BYREF
  char v45; // [rsp+240h] [rbp-D0h] BYREF

  v24 = sub_BD5C60(a2);
  v25 = &v33;
  v26 = &v34;
  v29 = 512;
  v19[0] = (unsigned int *)v20;
  v33 = &unk_49DA100;
  v19[1] = (unsigned int *)0x200000000LL;
  v23 = 0;
  v34 = &unk_49DA0B0;
  v27 = 0;
  v28 = 0;
  v30 = 7;
  v31 = 0;
  v32 = 0;
  v21 = 0;
  v22 = 0;
  sub_D5F1F0((__int64)v19, a2);
  sub_2895860((__int64)&v35, a1, a3, (__int64)&a8, (__int64)v19);
  v10 = a7;
  v40[1] = 0x1000000000LL;
  v40[0] = (unsigned __int64)v41;
  v11 = a4;
  if ( v36 )
  {
    sub_2894AD0((__int64)v40, (__int64)&v35, v9, (__int64)v40, a7, (__int64)&v35);
    v11 = a4;
    v10 = a7;
  }
  v12 = _mm_loadu_si128(&v38);
  v13 = *(_QWORD *)(a3 + 8);
  v43 = v39;
  v42 = v12;
  sub_289A510(&v44, a1, v13, (__int64)v40, v11, a5, a6, v10, (__int64)v19);
  sub_289E450(a1, a2, &v44, v19, v14, v15);
  if ( (char *)v44.m128i_i64[0] != &v45 )
    _libc_free(v44.m128i_u64[0]);
  if ( (_BYTE *)v40[0] != v41 )
    _libc_free(v40[0]);
  if ( v35 != &v37 )
    _libc_free((unsigned __int64)v35);
  nullsub_61();
  v33 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v19[0] != v20 )
    _libc_free((unsigned __int64)v19[0]);
}

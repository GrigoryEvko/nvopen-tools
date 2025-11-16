// Function: sub_26F96D0
// Address: 0x26f96d0
//
void __fastcall sub_26F96D0(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        _BYTE *a4,
        __int64 a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64, __int64),
        __int64 a8)
{
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int64 *v13; // rbx
  unsigned __int64 *v14; // r12
  unsigned __int64 v15; // rdi
  __int64 v16; // [rsp+0h] [rbp-310h]
  __int64 v17; // [rsp+0h] [rbp-310h]
  __int64 *v20; // [rsp+18h] [rbp-2F8h]
  __int64 v21; // [rsp+28h] [rbp-2E8h] BYREF
  __m128i v22; // [rsp+30h] [rbp-2E0h] BYREF
  _BYTE *v23[2]; // [rsp+40h] [rbp-2D0h] BYREF
  __int64 v24; // [rsp+50h] [rbp-2C0h] BYREF
  __int64 *v25; // [rsp+60h] [rbp-2B0h]
  __int64 v26; // [rsp+68h] [rbp-2A8h]
  __int64 v27; // [rsp+70h] [rbp-2A0h] BYREF
  __m128i v28; // [rsp+80h] [rbp-290h] BYREF
  _BYTE *v29[2]; // [rsp+90h] [rbp-280h] BYREF
  __int64 v30; // [rsp+A0h] [rbp-270h] BYREF
  __int64 *v31; // [rsp+B0h] [rbp-260h]
  __int64 v32; // [rsp+B8h] [rbp-258h]
  __int64 v33; // [rsp+C0h] [rbp-250h] BYREF
  __m128i v34; // [rsp+D0h] [rbp-240h] BYREF
  __int64 v35[2]; // [rsp+E0h] [rbp-230h] BYREF
  _QWORD v36[2]; // [rsp+F0h] [rbp-220h] BYREF
  __int64 v37[2]; // [rsp+100h] [rbp-210h] BYREF
  _QWORD v38[2]; // [rsp+110h] [rbp-200h] BYREF
  __m128i v39; // [rsp+120h] [rbp-1F0h]
  _QWORD v40[10]; // [rsp+130h] [rbp-1E0h] BYREF
  unsigned __int64 *v41; // [rsp+180h] [rbp-190h]
  unsigned int v42; // [rsp+188h] [rbp-188h]
  char v43; // [rsp+190h] [rbp-180h] BYREF

  v10 = sub_B491C0(*(_QWORD *)(a1 + 8));
  v11 = *(_QWORD *)(a1 + 8);
  v12 = *(_QWORD *)(v11 + 48);
  v21 = v12;
  if ( v12 )
  {
    v16 = v10;
    sub_B96E90((__int64)&v21, v12, 1);
    v11 = *(_QWORD *)(a1 + 8);
    v10 = v16;
  }
  v17 = *(_QWORD *)(v11 + 40);
  v20 = (__int64 *)a7(a8, v10);
  sub_B157E0((__int64)&v22, &v21);
  sub_B17430((__int64)v40, (__int64)"wholeprogramdevirt", (__int64)a2, a3, &v22, v17);
  sub_B16430((__int64)v23, "Optimization", 0xCu, a2, a3);
  v35[0] = (__int64)v36;
  sub_26F69E0(v35, v23[0], (__int64)&v23[0][(unsigned __int64)v23[1]]);
  v37[0] = (__int64)v38;
  sub_26F69E0(v37, v25, (__int64)v25 + v26);
  v39 = _mm_loadu_si128(&v28);
  sub_B180C0((__int64)v40, (unsigned __int64)v35);
  if ( (_QWORD *)v37[0] != v38 )
    j_j___libc_free_0(v37[0]);
  if ( (_QWORD *)v35[0] != v36 )
    j_j___libc_free_0(v35[0]);
  sub_B18290((__int64)v40, ": devirtualized a call to ", 0x1Au);
  sub_B16430((__int64)v29, "FunctionName", 0xCu, a4, a5);
  v35[0] = (__int64)v36;
  sub_26F69E0(v35, v29[0], (__int64)&v29[0][(unsigned __int64)v29[1]]);
  v37[0] = (__int64)v38;
  sub_26F69E0(v37, v31, (__int64)v31 + v32);
  v39 = _mm_loadu_si128(&v34);
  sub_B180C0((__int64)v40, (unsigned __int64)v35);
  if ( (_QWORD *)v37[0] != v38 )
    j_j___libc_free_0(v37[0]);
  if ( (_QWORD *)v35[0] != v36 )
    j_j___libc_free_0(v35[0]);
  sub_1049740(v20, (__int64)v40);
  if ( v31 != &v33 )
    j_j___libc_free_0((unsigned __int64)v31);
  if ( (__int64 *)v29[0] != &v30 )
    j_j___libc_free_0((unsigned __int64)v29[0]);
  if ( v25 != &v27 )
    j_j___libc_free_0((unsigned __int64)v25);
  if ( (__int64 *)v23[0] != &v24 )
    j_j___libc_free_0((unsigned __int64)v23[0]);
  v13 = v41;
  v40[0] = &unk_49D9D40;
  v14 = &v41[10 * v42];
  if ( v41 != v14 )
  {
    do
    {
      v14 -= 10;
      v15 = v14[4];
      if ( (unsigned __int64 *)v15 != v14 + 6 )
        j_j___libc_free_0(v15);
      if ( (unsigned __int64 *)*v14 != v14 + 2 )
        j_j___libc_free_0(*v14);
    }
    while ( v13 != v14 );
    v14 = v41;
  }
  if ( v14 != (unsigned __int64 *)&v43 )
    _libc_free((unsigned __int64)v14);
  if ( v21 )
    sub_B91220((__int64)&v21, v21);
}

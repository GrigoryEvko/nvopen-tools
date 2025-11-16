// Function: sub_2DE1C70
// Address: 0x2de1c70
//
void __fastcall sub_2DE1C70(__int8 *a1, size_t a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v12; // [rsp+10h] [rbp-210h]
  __int64 v13; // [rsp+28h] [rbp-1F8h] BYREF
  __m128i v14; // [rsp+30h] [rbp-1F0h] BYREF
  _QWORD v15[10]; // [rsp+40h] [rbp-1E0h] BYREF
  unsigned __int64 *v16; // [rsp+90h] [rbp-190h]
  unsigned int v17; // [rsp+98h] [rbp-188h]
  char v18; // [rsp+A0h] [rbp-180h] BYREF

  v12 = **(_QWORD **)(a6 + 32);
  sub_D4BD20(&v13, a6, a3, a4, (__int64)a5, v12);
  sub_B157E0((__int64)&v14, &v13);
  sub_B17850((__int64)v15, (__int64)"hardware-loops", a3, a4, &v14, v12);
  sub_B18290((__int64)v15, "hardware-loop not created: ", 0x1Bu);
  if ( v13 )
    sub_B91220((__int64)&v13, v13);
  sub_B18290((__int64)v15, a1, a2);
  sub_1049740(a5, (__int64)v15);
  v7 = v16;
  v15[0] = &unk_49D9D40;
  v8 = &v16[10 * v17];
  if ( v16 != v8 )
  {
    do
    {
      v8 -= 10;
      v9 = v8[4];
      if ( (unsigned __int64 *)v9 != v8 + 6 )
        j_j___libc_free_0(v9);
      if ( (unsigned __int64 *)*v8 != v8 + 2 )
        j_j___libc_free_0(*v8);
    }
    while ( v7 != v8 );
    v8 = v16;
  }
  if ( v8 != (unsigned __int64 *)&v18 )
    _libc_free((unsigned __int64)v8);
}

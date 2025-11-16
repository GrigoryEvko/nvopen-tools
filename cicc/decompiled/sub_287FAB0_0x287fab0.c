// Function: sub_287FAB0
// Address: 0x287fab0
//
void __fastcall sub_287FAB0(__int64 *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r12
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  __m128i v12; // [rsp+0h] [rbp-1F0h] BYREF
  _QWORD v13[10]; // [rsp+10h] [rbp-1E0h] BYREF
  unsigned __int64 *v14; // [rsp+60h] [rbp-190h]
  unsigned int v15; // [rsp+68h] [rbp-188h]
  char v16; // [rsp+70h] [rbp-180h] BYREF

  v4 = *a1;
  v5 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v5)
    || (v10 = sub_B2BE50(v4),
        v11 = sub_B6F970(v10),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v11 + 48LL))(v11)) )
  {
    v6 = *a3;
    sub_B157E0((__int64)&v12, a2);
    sub_B17430((__int64)v13, (__int64)"loop-unroll", (__int64)"computeUnrollCount", 18, &v12, v6);
    sub_B18290((__int64)v13, "      Failed: not an innermost loop", 0x23u);
    sub_1049740(a1, (__int64)v13);
    v7 = v14;
    v13[0] = &unk_49D9D40;
    v8 = &v14[10 * v15];
    if ( v14 != v8 )
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
      v8 = v14;
    }
    if ( v8 != (unsigned __int64 *)&v16 )
      _libc_free((unsigned __int64)v8);
  }
}

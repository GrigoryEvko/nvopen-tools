// Function: sub_287FD70
// Address: 0x287fd70
//
void __fastcall sub_287FD70(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rsi
  __int64 v9; // r15
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-1F8h] BYREF
  __m128i v16; // [rsp+10h] [rbp-1F0h] BYREF
  _QWORD v17[10]; // [rsp+20h] [rbp-1E0h] BYREF
  unsigned __int64 *v18; // [rsp+70h] [rbp-190h]
  unsigned int v19; // [rsp+78h] [rbp-188h]
  char v20; // [rsp+80h] [rbp-180h] BYREF

  v2 = *a1;
  v3 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v3)
    || (v13 = sub_B2BE50(v2),
        v14 = sub_B6F970(v13),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v14 + 48LL))(v14)) )
  {
    v8 = *a2;
    v9 = **(_QWORD **)(v8 + 32);
    sub_D4BD20(&v15, v8, v4, v5, v6, v7);
    sub_B157E0((__int64)&v16, &v15);
    sub_B17430((__int64)v17, (__int64)"loop-unroll", (__int64)"profitableToRuntimeUnroll", 25, &v16, v9);
    if ( v15 )
      sub_B91220((__int64)&v15, v15);
    sub_B18290((__int64)v17, "      Failed: Not innermost loop", 0x20u);
    sub_1049740(a1, (__int64)v17);
    v10 = v18;
    v17[0] = &unk_49D9D40;
    v11 = &v18[10 * v19];
    if ( v18 != v11 )
    {
      do
      {
        v11 -= 10;
        v12 = v11[4];
        if ( (unsigned __int64 *)v12 != v11 + 6 )
          j_j___libc_free_0(v12);
        if ( (unsigned __int64 *)*v11 != v11 + 2 )
          j_j___libc_free_0(*v11);
      }
      while ( v10 != v11 );
      v11 = v18;
    }
    if ( v11 != (unsigned __int64 *)&v20 )
      _libc_free((unsigned __int64)v11);
  }
}

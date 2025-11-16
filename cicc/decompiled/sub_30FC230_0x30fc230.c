// Function: sub_30FC230
// Address: 0x30fc230
//
void __fastcall sub_30FC230(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // r15
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __m128i v10; // [rsp+0h] [rbp-1F0h] BYREF
  _QWORD v11[10]; // [rsp+10h] [rbp-1E0h] BYREF
  unsigned __int64 *v12; // [rsp+60h] [rbp-190h]
  unsigned int v13; // [rsp+68h] [rbp-188h]
  char v14; // [rsp+70h] [rbp-180h] BYREF

  v2 = *a1;
  v3 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v3)
    || (v8 = sub_B2BE50(v2), v9 = sub_B6F970(v8),
                             (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v9 + 48LL))(v9)) )
  {
    v4 = *(_QWORD *)(a2 + 40);
    sub_B157E0((__int64)&v10, (_QWORD *)(a2 + 32));
    sub_B17640((__int64)v11, (__int64)"inline-ml", (__int64)"InliningAttemptedAndUnsuccessful", 32, &v10, v4);
    sub_30FBC60(a2, (__int64)v11);
    sub_1049740(a1, (__int64)v11);
    v5 = v12;
    v11[0] = &unk_49D9D40;
    v6 = &v12[10 * v13];
    if ( v12 != v6 )
    {
      do
      {
        v6 -= 10;
        v7 = v6[4];
        if ( (unsigned __int64 *)v7 != v6 + 6 )
          j_j___libc_free_0(v7);
        if ( (unsigned __int64 *)*v6 != v6 + 2 )
          j_j___libc_free_0(*v6);
      }
      while ( v5 != v6 );
      v6 = v12;
    }
    if ( v6 != (unsigned __int64 *)&v14 )
      _libc_free((unsigned __int64)v6);
  }
}

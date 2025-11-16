// Function: sub_2434B50
// Address: 0x2434b50
//
void __fastcall sub_2434B50(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  unsigned __int64 *v4; // rbx
  unsigned __int64 *v5; // r12
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD v9[10]; // [rsp+0h] [rbp-1D0h] BYREF
  unsigned __int64 *v10; // [rsp+50h] [rbp-180h]
  unsigned int v11; // [rsp+58h] [rbp-178h]
  char v12; // [rsp+60h] [rbp-170h] BYREF

  v2 = *a1;
  v3 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v3)
    || (v7 = sub_B2BE50(v2), v8 = sub_B6F970(v7),
                             (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v8 + 48LL))(v8)) )
  {
    sub_B174A0((__int64)v9, (__int64)"hwasan", (__int64)"ignoreAccess", 12, *a2);
    sub_1049740(a1, (__int64)v9);
    v4 = v10;
    v9[0] = &unk_49D9D40;
    v5 = &v10[10 * v11];
    if ( v10 != v5 )
    {
      do
      {
        v5 -= 10;
        v6 = v5[4];
        if ( (unsigned __int64 *)v6 != v5 + 6 )
          j_j___libc_free_0(v6);
        if ( (unsigned __int64 *)*v5 != v5 + 2 )
          j_j___libc_free_0(*v5);
      }
      while ( v4 != v5 );
      v5 = v10;
    }
    if ( v5 != (unsigned __int64 *)&v12 )
      _libc_free((unsigned __int64)v5);
  }
}

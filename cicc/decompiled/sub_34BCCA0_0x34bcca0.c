// Function: sub_34BCCA0
// Address: 0x34bcca0
//
void __fastcall sub_34BCCA0(
        unsigned __int64 *a1,
        unsigned __int8 (__fastcall *a2)(__int64, __int64 *, unsigned __int64 *),
        __int64 a3)
{
  unsigned __int64 v3; // rcx
  unsigned __int64 *i; // rdx
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // rax
  unsigned __int64 v8; // [rsp-38h] [rbp-38h] BYREF
  unsigned __int64 *v9; // [rsp-30h] [rbp-30h]

  v3 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (unsigned __int64 *)v3 != a1 )
  {
    i = (unsigned __int64 *)a1[1];
    v6 = (unsigned __int64 *)i[1];
    if ( a1 != v6 )
    {
      if ( a1 != i )
      {
        for ( i = (unsigned __int64 *)i[1]; ; i = (unsigned __int64 *)i[1] )
        {
          v7 = (unsigned __int64 *)v6[1];
          if ( a1 == v7 )
            break;
          v6 = (unsigned __int64 *)v7[1];
          if ( a1 == v6 )
            break;
        }
      }
      v9 = &v8;
      v8 = (unsigned __int64)&v8 + 4;
      if ( a1 != i )
      {
        *(_QWORD *)((*i & 0xFFFFFFFFFFFFFFF8LL) + 8) = a1;
        *a1 = *a1 & 7 | *i & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v3 + 8) = &v8;
        *i = (unsigned __int64)&v8 | *i & 7;
        v9 = i;
        v8 = v8 & 7 | v3;
      }
      sub_34BCCA0(a1, a2, a3);
      sub_34BCCA0(&v8, a2, a3);
      sub_34BCB10(a1, &v8, a2, a3);
    }
  }
}

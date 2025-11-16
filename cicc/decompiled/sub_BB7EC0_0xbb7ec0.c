// Function: sub_BB7EC0
// Address: 0xbb7ec0
//
__int64 __fastcall sub_BB7EC0(__int64 *a1, __int64 *a2, int a3)
{
  __int64 v4; // r12
  void (__fastcall *v5)(__int64, __int64, __int64); // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r12
  void (__fastcall *v9)(__int64, __int64, __int64); // rax

  if ( a3 == 1 )
  {
    *a1 = *a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v4 = *a1;
      if ( *a1 )
      {
        v5 = *(void (__fastcall **)(__int64, __int64, __int64))(v4 + 16);
        if ( v5 )
          v5(*a1, *a1, 3);
        j_j___libc_free_0(v4, 32);
      }
    }
    return 0;
  }
  v6 = *a2;
  v7 = sub_22077B0(32);
  v8 = v7;
  if ( v7 )
  {
    *(_QWORD *)(v7 + 16) = 0;
    v9 = *(void (__fastcall **)(__int64, __int64, __int64))(v6 + 16);
    if ( v9 )
    {
      v9(v8, v6, 2);
      *(_QWORD *)(v8 + 24) = *(_QWORD *)(v6 + 24);
      *(_QWORD *)(v8 + 16) = *(_QWORD *)(v6 + 16);
    }
  }
  *a1 = v8;
  return 0;
}

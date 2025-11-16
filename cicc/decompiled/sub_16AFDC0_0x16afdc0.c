// Function: sub_16AFDC0
// Address: 0x16afdc0
//
unsigned __int64 __fastcall sub_16AFDC0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  __int64 (*v3)(void); // rax
  int v4; // r13d
  unsigned int i; // r14d
  __int64 v6; // rdx
  unsigned __int64 v7; // rdx
  int v9; // r13d
  unsigned int j; // r14d
  __int64 v11; // rdx
  unsigned __int64 v12; // rdx

  v2 = *(_QWORD *)(a2 + 32);
  v3 = *(__int64 (**)(void))(*(_QWORD *)a1 + 16LL);
  if ( v2 )
  {
    v2 += 6LL;
    v4 = v3();
    if ( v4 )
    {
      for ( i = 0; i != v4; ++i )
      {
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, i);
        v7 = v6 + 8;
        if ( v2 < v7 )
          v2 = v7;
      }
    }
    return v2;
  }
  v9 = v3();
  if ( !v9 )
    return v2;
  for ( j = 0; j != v9; ++j )
  {
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 24LL))(a1, j);
    v12 = v11 + 8;
    if ( v2 < v12 )
      v2 = v12;
  }
  return v2;
}

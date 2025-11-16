// Function: sub_1E69BA0
// Address: 0x1e69ba0
//
void __fastcall sub_1E69BA0(_QWORD *a1, int a2, int a3)
{
  __int64 v3; // r13
  __int64 (*v5)(void); // rax
  __int64 v6; // rbx
  __int64 v7; // rdi

  v3 = 0;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(*a1 + 16LL) + 112LL);
  if ( v5 != sub_1D00B10 )
    v3 = v5();
  if ( a2 < 0 )
    v6 = *(_QWORD *)(a1[3] + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v6 = *(_QWORD *)(a1[34] + 8LL * (unsigned int)a2);
  while ( v6 )
  {
    while ( 1 )
    {
      v7 = v6;
      v6 = *(_QWORD *)(v6 + 32);
      if ( a3 <= 0 )
        break;
      sub_1E311F0(v7, (unsigned int)a3, v3);
      if ( !v6 )
        return;
    }
    sub_1E310D0(v7, a3);
  }
}

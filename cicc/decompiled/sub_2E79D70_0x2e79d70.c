// Function: sub_2E79D70
// Address: 0x2e79d70
//
__int64 __fastcall sub_2E79D70(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  unsigned int v7; // r8d
  _QWORD *v8; // rax
  _QWORD *v9; // rcx

  v6 = *(_QWORD *)(a1 + 8) + 32LL * a2;
  v7 = 0;
  v8 = *(_QWORD **)v6;
  v9 = *(_QWORD **)(v6 + 8);
  if ( *(_QWORD **)v6 != v9 )
  {
    do
    {
      while ( *v8 != a3 )
      {
        if ( ++v8 == v9 )
          return v7;
      }
      *v8++ = a4;
      v7 = 1;
    }
    while ( v8 != v9 );
  }
  return v7;
}

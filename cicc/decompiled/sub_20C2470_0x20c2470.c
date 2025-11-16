// Function: sub_20C2470
// Address: 0x20c2470
//
__int64 __fastcall sub_20C2470(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // r9
  __int64 v6; // rdi
  unsigned int v7; // edx
  __int64 result; // rax
  unsigned __int64 v9; // rsi
  int v10; // ecx

  v3 = a1[4];
  v6 = a1[1];
  v7 = *(_DWORD *)(v3 + 4LL * a2);
  do
  {
    result = v7;
    v7 = *(_DWORD *)(v6 + 4LL * v7);
  }
  while ( (_DWORD)result != v7 );
  LODWORD(v9) = *(_DWORD *)(v3 + 4LL * a3);
  do
  {
    v10 = v9;
    v9 = *(unsigned int *)(v6 + 4LL * (unsigned int)v9);
  }
  while ( v10 != (_DWORD)v9 );
  if ( (_DWORD)result )
  {
    result = (unsigned int)v9;
    v9 = v7;
  }
  if ( v9 >= (a1[2] - v6) >> 2 )
    sub_222CF80("vector::_M_range_check: __n (which is %zu) >= this->size() (which is %zu)", v9);
  *(_DWORD *)(v6 + 4 * v9) = result;
  return result;
}

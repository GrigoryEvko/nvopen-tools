// Function: sub_16876E0
// Address: 0x16876e0
//
__int64 __fastcall sub_16876E0(__int64 a1, void (__fastcall *a2)(_QWORD, __int64), __int64 a3)
{
  __int64 result; // rax
  unsigned int v5; // r13d
  unsigned int v6; // ebx
  unsigned int v7; // ecx
  int v8; // r12d
  __int64 v9; // [rsp+8h] [rbp-48h]

  if ( *(_QWORD *)(a1 + 48) )
  {
    result = *(unsigned int *)(a1 + 80);
    if ( (_DWORD)result )
    {
      v9 = 0;
      do
      {
        v5 = *(_DWORD *)(*(_QWORD *)(a1 + 96) + 4 * v9);
        if ( v5 )
        {
          do
          {
            v6 = v5;
            _BitScanForward(&v7, v5);
            v8 = 1 << v7;
            v5 ^= 1 << v7;
            a2(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 8LL * (v7 + 32 * (_DWORD)v9)), a3);
          }
          while ( v8 != v6 );
        }
        result = ++v9;
      }
      while ( *(_DWORD *)(a1 + 80) > (unsigned int)v9 );
    }
  }
  return result;
}

// Function: sub_130DB10
// Address: 0x130db10
//
__int64 __fastcall sub_130DB10(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // rbx
  _QWORD *v6; // rsi
  __int64 result; // rax
  __int64 v8; // [rsp-48h] [rbp-48h]
  unsigned __int64 v9; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v5 = 0;
    v9 = (a3 >> 1) & 0xFFFFFFFFFFFFFFF8LL;
    v8 = a3 - 8;
    do
    {
      v6 = *(_QWORD **)(a1 + 8 * v5);
      if ( *v6 != 0x5B5B5B5B5B5B5B5BLL
        || *(_QWORD *)((char *)v6 + v9) != 0x5B5B5B5B5B5B5B5BLL
        || (result = v8, *(_QWORD *)((char *)v6 + v8) != 0x5B5B5B5B5B5B5B5BLL) )
      {
        result = sub_130D560("<jemalloc>: Write-after-free detected on deallocated pointer %p (size %zu).\n", v6, a3);
      }
      ++v5;
    }
    while ( a2 != v5 );
  }
  return result;
}

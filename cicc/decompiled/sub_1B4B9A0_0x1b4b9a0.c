// Function: sub_1B4B9A0
// Address: 0x1b4b9a0
//
__int64 __fastcall sub_1B4B9A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // rdx
  unsigned __int64 v9; // rbx
  __int64 result; // rax
  __int64 v11; // rcx

  v6 = a3 - a2;
  v7 = a3 - a2;
  v8 = *(unsigned int *)(a1 + 8);
  v9 = v7 >> 3;
  if ( (unsigned __int64)*(unsigned int *)(a1 + 12) - v8 < v9 )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), v9 + v8, 4, a5, a6);
    v8 = *(unsigned int *)(a1 + 8);
  }
  result = *(_QWORD *)a1;
  v11 = *(_QWORD *)a1 + 4 * v8;
  if ( v6 > 0 )
  {
    result = 0;
    do
    {
      *(_DWORD *)(v11 + 4 * result) = *(_QWORD *)(a2 + 8 * result);
      ++result;
    }
    while ( (__int64)(v9 - result) > 0 );
    LODWORD(v8) = *(_DWORD *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = v8 + v9;
  return result;
}

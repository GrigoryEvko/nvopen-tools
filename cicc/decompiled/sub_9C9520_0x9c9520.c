// Function: sub_9C9520
// Address: 0x9c9520
//
__int64 __fastcall sub_9C9520(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // rax

  result = *(unsigned int *)(a1 + 8);
  v4 = a3 - a2;
  v5 = (a3 - a2) >> 3;
  if ( v5 + result > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, v5 + result, 4);
    result = *(unsigned int *)(a1 + 8);
  }
  v6 = *(_QWORD *)a1 + 4 * result;
  if ( v4 > 0 )
  {
    v7 = 0;
    do
    {
      *(_DWORD *)(v6 + 4 * v7) = *(_QWORD *)(a2 + 8 * v7);
      ++v7;
    }
    while ( v5 - v7 > 0 );
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = result + v5;
  return result;
}

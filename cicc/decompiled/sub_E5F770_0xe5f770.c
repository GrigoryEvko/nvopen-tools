// Function: sub_E5F770
// Address: 0xe5f770
//
__int64 __fastcall sub_E5F770(__int64 a1, int a2)
{
  __int64 v2; // rsi
  __int64 result; // rax

  v2 = (unsigned int)(a2 - 1);
  result = 0;
  if ( (unsigned int)v2 < *(_DWORD *)(a1 + 72) )
    return *(unsigned __int8 *)(*(_QWORD *)(a1 + 64) + 32 * v2 + 4);
  return result;
}

// Function: sub_38D6650
// Address: 0x38d6650
//
__int64 __fastcall sub_38D6650(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  int v6; // r8d
  int v7; // r9d

  sub_38DC4E0(a1, a2, a3);
  result = sub_390D5F0(*(_QWORD *)(a1 + 264), a2, 0);
  if ( a4 && *(_BYTE *)(a4 + 16) == 1 )
  {
    *a2 = *a2 & 7 | a4;
  }
  else
  {
    result = *(unsigned int *)(a1 + 296);
    if ( (unsigned int)result >= *(_DWORD *)(a1 + 300) )
    {
      sub_16CD150(a1 + 288, (const void *)(a1 + 304), 0, 8, v6, v7);
      result = *(unsigned int *)(a1 + 296);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 288) + 8 * result) = a2;
    ++*(_DWORD *)(a1 + 296);
  }
  return result;
}

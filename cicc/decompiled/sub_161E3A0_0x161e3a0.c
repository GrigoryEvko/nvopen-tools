// Function: sub_161E3A0
// Address: 0x161e3a0
//
__int64 __fastcall sub_161E3A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 result; // rax

  v4 = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)v4 )
  {
    result = sub_161DF70((__int64 *)a1, a2, a3);
    if ( (_BYTE)result )
      return result;
    v4 = *(unsigned int *)(a1 + 8);
  }
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v4 )
  {
    sub_16CD150(a1, a1 + 16, 0, 8);
    v4 = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v4) = a2;
  result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = result;
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)result )
  {
    sub_16CD150(a1, a1 + 16, 0, 8);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = a3;
  ++*(_DWORD *)(a1 + 8);
  return result;
}

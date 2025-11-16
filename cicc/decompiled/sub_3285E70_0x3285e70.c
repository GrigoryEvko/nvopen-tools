// Function: sub_3285E70
// Address: 0x3285e70
//
__int64 __fastcall sub_3285E70(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 result; // rax

  v3 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)a1 = v3;
  if ( v3 )
    sub_B96E90(a1, v3, 1);
  result = *(unsigned int *)(a2 + 72);
  *(_DWORD *)(a1 + 8) = result;
  return result;
}

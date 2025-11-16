// Function: sub_14A9240
// Address: 0x14a9240
//
__int64 __fastcall sub_14A9240(__int64 a1, __int64 *a2)
{
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 8) > 0x40u )
    return sub_16A8890(a1, a2);
  result = *a2;
  *(_QWORD *)a1 &= *a2;
  return result;
}

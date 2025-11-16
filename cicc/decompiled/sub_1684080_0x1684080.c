// Function: sub_1684080
// Address: 0x1684080
//
__int64 __fastcall sub_1684080(__int64 (__fastcall *a1)(), __int64 (__fastcall *a2)(), unsigned int a3)
{
  __int64 result; // rax

  result = sub_1683D70(a3);
  *(_QWORD *)result = a1;
  *(_QWORD *)(result + 8) = a2;
  if ( a2 == sub_16881E0 && a1 == sub_16881D0 )
    *(_WORD *)(result + 84) = *(_WORD *)(result + 84) & 0xF00F | 0x20;
  if ( a2 == sub_1688220 && a1 == sub_1688200 )
    *(_WORD *)(result + 84) = *(_WORD *)(result + 84) & 0xF00F | 0x10;
  return result;
}

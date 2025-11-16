// Function: sub_1687490
// Address: 0x1687490
//
_QWORD *__fastcall sub_1687490(__int64 (__fastcall *a1)(), __int64 (__fastcall *a2)(), unsigned int a3)
{
  _QWORD *result; // rax

  result = sub_1687180(a3, (__int64)a2);
  *result = a1;
  result[1] = a2;
  if ( a2 == sub_16881E0 && a1 == sub_16881D0 )
    *((_WORD *)result + 42) = *((_WORD *)result + 42) & 0xF00F | 0x20;
  if ( a2 == sub_1688220 && a1 == sub_1688200 )
    *((_WORD *)result + 42) = *((_WORD *)result + 42) & 0xF00F | 0x10;
  return result;
}

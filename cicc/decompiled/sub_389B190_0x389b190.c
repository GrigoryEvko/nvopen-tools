// Function: sub_389B190
// Address: 0x389b190
//
__int64 *__fastcall sub_389B190(__int64 *a1, unsigned int a2, unsigned __int64 a3)
{
  _QWORD *v4; // rax
  __int64 v5; // rax
  __int64 *result; // rax

  v4 = (_QWORD *)sub_15E0530(a1[1]);
  v5 = sub_1643280(v4);
  result = sub_389ACD0(a1, a2, v5, a3, 0);
  if ( result )
  {
    if ( *((_BYTE *)result + 16) != 18 )
      return 0;
  }
  return result;
}

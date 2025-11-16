// Function: sub_16B7CB0
// Address: 0x16b7cb0
//
__int64 __fastcall sub_16B7CB0(int a1, const void *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9

  v5 = a5;
  if ( !qword_4FA01E0 )
  {
    sub_16C1EA0(&qword_4FA01E0, sub_16B89A0, sub_16B0D50);
    v5 = a5;
  }
  return sub_16B5B30(qword_4FA01E0, a1, a2, a3, a4, v5);
}

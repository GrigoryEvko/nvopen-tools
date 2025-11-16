// Function: sub_1308810
// Address: 0x1308810
//
__int64 __fastcall sub_1308810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  unsigned __int64 v5; // r15
  __int64 v7; // rax
  _BYTE v8[80]; // [rsp+10h] [rbp-50h] BYREF

  v3 = a2;
  if ( unk_4F96B58 )
  {
    v5 = __readfsqword(0) - 2664;
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v7 = sub_1313D30(v5, 0);
      v3 = a2;
      v5 = v7;
    }
  }
  else
  {
    v5 = 0;
  }
  sub_131D1C0(v5, v8, a1, v3, 0, 0x10000);
  sub_130F6A0(&sub_131D290, v8, a3);
  return sub_131D350(v5, v8);
}

// Function: sub_D13FF0
// Address: 0xd13ff0
//
__int64 __fastcall sub_D13FF0(__int64 a1, char a2, __int64 a3, __int64 a4, char a5, int a6, __int64 a7)
{
  unsigned int v7; // r13d
  _QWORD v9[3]; // [rsp+0h] [rbp-50h] BYREF
  char v10; // [rsp+18h] [rbp-38h]
  char v11; // [rsp+19h] [rbp-37h]
  unsigned __int8 v12; // [rsp+1Ah] [rbp-36h]
  __int64 v13; // [rsp+20h] [rbp-30h]

  if ( !a4 )
    return sub_D13FA0(a1, a2, a6);
  v10 = a2;
  v9[1] = a3;
  v9[0] = off_49DDDD0;
  v9[2] = a4;
  v11 = a5;
  v13 = a7;
  v12 = 0;
  sub_D13D60(a1, (__int64)v9, a6);
  v7 = v12;
  v9[0] = off_49DDDD0;
  nullsub_185();
  return v7;
}

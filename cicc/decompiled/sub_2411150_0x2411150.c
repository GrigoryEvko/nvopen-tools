// Function: sub_2411150
// Address: 0x2411150
//
__int64 __fastcall sub_2411150(__int64 a1, __int64 a2, unsigned int **a3)
{
  __int64 v5; // [rsp+8h] [rbp-48h] BYREF
  _BYTE v6[32]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v7; // [rsp+30h] [rbp-20h]

  v5 = a2;
  if ( !byte_4FE2E28 && (unsigned int)sub_2207590((__int64)&byte_4FE2E28) )
  {
    byte_4FE2E30 = (_DWORD)qword_4FE2FA8 != 0;
    sub_2207640((__int64)&byte_4FE2E28);
  }
  if ( !byte_4FE2E30 )
    return v5;
  v7 = 257;
  return sub_921880(
           a3,
           *(_QWORD *)(*(_QWORD *)a1 + 520LL),
           *(_QWORD *)(*(_QWORD *)a1 + 528LL),
           (int)&v5,
           1,
           (__int64)v6,
           0);
}

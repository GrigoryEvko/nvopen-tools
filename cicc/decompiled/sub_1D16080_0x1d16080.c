// Function: sub_1D16080
// Address: 0x1d16080
//
__int64 __fastcall sub_1D16080(unsigned int *a1, __int64 a2, __int64 a3)
{
  char v3; // r12
  _QWORD v5[6]; // [rsp+0h] [rbp-30h] BYREF

  v5[0] = a2;
  v5[1] = a3;
  if ( (_BYTE)a2 )
  {
    if ( (unsigned __int8)(a2 - 14) > 0x5Fu )
    {
      v3 = (unsigned __int8)(a2 - 86) <= 0x17u || (unsigned __int8)(a2 - 8) <= 5u;
      goto LABEL_4;
    }
    return a1[17];
  }
  v3 = sub_1F58CD0(v5);
  if ( (unsigned __int8)sub_1F58D20(v5) )
    return a1[17];
LABEL_4:
  if ( v3 )
    return a1[16];
  else
    return a1[15];
}

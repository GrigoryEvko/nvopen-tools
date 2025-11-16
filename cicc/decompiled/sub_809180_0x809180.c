// Function: sub_809180
// Address: 0x809180
//
__int64 __fastcall sub_809180(signed __int64 a1, _QWORD *a2)
{
  __int64 v2; // rdx
  char v4; // [rsp+10h] [rbp-50h] BYREF
  _BYTE v5[71]; // [rsp+11h] [rbp-4Fh] BYREF

  if ( a1 < 0 )
  {
    v4 = 45;
    if ( a1 >= -9 )
    {
      v5[1] = 0;
      v2 = 2;
      v5[0] = 48 - a1;
LABEL_7:
      v4 = 110;
      goto LABEL_4;
    }
    v2 = (int)(sub_622470(-a1, v5) + 1);
LABEL_9:
    if ( v4 != 45 )
      goto LABEL_4;
    goto LABEL_7;
  }
  if ( a1 > 9 )
  {
    v2 = (int)sub_622470(a1, &v4);
    goto LABEL_9;
  }
  v5[0] = 0;
  v2 = 1;
  v4 = a1 + 48;
LABEL_4:
  *a2 += v2;
  return sub_8238B0(qword_4F18BE0, &v4, v2);
}

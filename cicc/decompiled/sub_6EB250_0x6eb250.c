// Function: sub_6EB250
// Address: 0x6eb250
//
__int64 __fastcall sub_6EB250(int a1, int a2, int a3, int a4, __int64 a5)
{
  int v8; // eax
  int v9; // r9d
  char v10; // cl
  int v11; // r8d

  v8 = sub_6E6010();
  if ( a5 )
  {
    v9 = 0;
    if ( !qword_4D03C50 )
    {
LABEL_6:
      v11 = 0;
      return sub_87CD50(a1, a2, a3, a4, v11, v9, v8, a5);
    }
    goto LABEL_3;
  }
  if ( !qword_4D03C50 )
  {
    v11 = 0;
    v9 = 1;
    return sub_87CD50(a1, a2, a3, a4, v11, v9, v8, a5);
  }
  v10 = *(_BYTE *)(qword_4D03C50 + 17LL);
  v9 = 1;
  if ( (v10 & 0x40) != 0 )
  {
LABEL_3:
    v10 = *(_BYTE *)(qword_4D03C50 + 17LL);
    v9 = 0;
  }
  v11 = 1;
  if ( (v10 & 2) == 0 )
    goto LABEL_6;
  return sub_87CD50(a1, a2, a3, a4, v11, v9, v8, a5);
}

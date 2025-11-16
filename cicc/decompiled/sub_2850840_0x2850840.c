// Function: sub_2850840
// Address: 0x2850840
//
char __fastcall sub_2850840(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        char a6,
        unsigned __int8 a7)
{
  char result; // al
  unsigned __int64 v9; // r13
  __int64 v10; // rax
  char v11; // al
  char v12; // r8
  char v13; // [rsp+Fh] [rbp-31h]
  unsigned int v14; // [rsp+10h] [rbp-30h]
  __int64 v15; // [rsp+18h] [rbp-28h]

  result = 1;
  if ( a5 )
  {
    if ( a2 == 3 )
    {
      v10 = a7;
      v9 = -1;
    }
    else
    {
      v9 = a7;
      v10 = 1;
      if ( a3 )
      {
        v13 = a6;
        v14 = a4;
        v15 = a3;
        v11 = sub_BCEA30(a3);
        a3 = v15;
        v12 = v11;
        a4 = v14;
        a6 = v13;
        v10 = 1;
        if ( v12 )
        {
          if ( (_BYTE)qword_5000F88 )
            v9 = 0;
        }
      }
    }
    return sub_2850560(a1, a2, a3, a4, 0, a5, a6, v10, v9);
  }
  return result;
}

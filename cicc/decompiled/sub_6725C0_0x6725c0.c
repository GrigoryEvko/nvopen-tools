// Function: sub_6725C0
// Address: 0x6725c0
//
_BOOL8 sub_6725C0()
{
  __int64 v0; // rdx
  __int64 v1; // rcx
  _BOOL4 v2; // r13d
  int v4; // r13d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  _BYTE v11[64]; // [rsp+0h] [rbp-40h] BYREF

  sub_7ADF70(v11, 0);
  sub_7AE360(v11);
  sub_7B8B50(v11, 0, v0, v1);
  sub_672540((__int64)v11, 0);
  if ( word_4F06418[0] != 27 )
  {
    if ( word_4F06418[0] == 1 )
    {
      sub_7AE360(v11);
      sub_7B8B50(v11, 0, v7, v8);
      sub_672540((__int64)v11, 0);
      if ( word_4F06418[0] == 27 && !(unsigned int)sub_7C6040(v11, 0) )
      {
        sub_7AE360(v11);
        sub_7B8B50(v11, 0, v9, v10);
        sub_672540((__int64)v11, 0);
        v2 = word_4F06418[0] == 30;
        goto LABEL_4;
      }
    }
    goto LABEL_3;
  }
  v4 = 2;
  if ( (unsigned int)sub_7C6040(v11, 0) )
  {
LABEL_3:
    v2 = 0;
    goto LABEL_4;
  }
  while ( 1 )
  {
    sub_7AE360(v11);
    sub_7B8B50(v11, 0, v5, v6);
    sub_672540((__int64)v11, 0);
    if ( word_4F06418[0] == 30 )
      break;
    if ( word_4F06418[0] == 27 && v4 != 1 )
    {
      v4 = 1;
      if ( !(unsigned int)sub_7C6040(v11, 0) )
        continue;
    }
    goto LABEL_3;
  }
  v2 = 1;
LABEL_4:
  sub_7BC000(v11);
  return v2;
}

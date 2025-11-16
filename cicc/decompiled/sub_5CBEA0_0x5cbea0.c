// Function: sub_5CBEA0
// Address: 0x5cbea0
//
__int64 __fastcall sub_5CBEA0(char a1, char a2, __int16 a3, __int64 a4)
{
  __int64 *v5; // r12
  __int64 v6; // rax
  int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v14[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = &v13;
  v13 = 0;
  while ( 1 )
  {
    if ( (unsigned int)sub_869470(v14) )
    {
      do
      {
        if ( word_4F06418[0] == a3 || word_4F06418[0] == 67 )
        {
          v8 = sub_727670();
          *(_BYTE *)(v8 + 9) = a2;
          *(_QWORD *)(v8 + 56) = unk_4F063F8;
          *v5 = v8;
          v9 = unk_4F063F8;
          *(_BYTE *)(v8 + 8) = 1;
          *(_QWORD *)(*v5 + 64) = v9;
        }
        else
        {
          *v5 = sub_5CBA50(a2, a4);
        }
        v6 = sub_867630(v14[0], 0);
        if ( v6 && *v5 )
        {
          *(_BYTE *)(*v5 + 11) |= 8u;
          *(_QWORD *)(*v5 + 72) = v6;
        }
        v7 = sub_866C00(v14[0]);
        if ( *v5 )
          v5 = (__int64 *)sub_5CB9F0((_QWORD **)v5);
      }
      while ( v7 );
    }
    if ( word_4F06418[0] == a3 )
      break;
    if ( word_4F06418[0] != 7 && word_4F06418[0] != 1 || a2 != 3 )
    {
      if ( word_4F06418[0] != 67 )
      {
        ++*(_BYTE *)(unk_4F061C8 + 75LL);
        sub_6851D0(253);
        --*(_BYTE *)(unk_4F061C8 + 75LL);
        if ( word_4F06418[0] != 67 )
          break;
      }
      sub_7B8B50();
    }
  }
  v10 = v13;
  if ( v13 )
  {
    do
    {
      *(_BYTE *)(v10 + 10) = a1;
      sub_5CB870(v10);
      v10 = *(_QWORD *)v10;
    }
    while ( v10 );
    return v13;
  }
  return v10;
}

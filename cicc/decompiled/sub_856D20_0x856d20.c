// Function: sub_856D20
// Address: 0x856d20
//
__int64 __fastcall sub_856D20(unsigned __int64 a1, unsigned int *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 result; // rax
  int v7; // r13d
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9

  if ( qword_4D03CD8 <= qword_4D03CD0 )
  {
    sub_6851C0(0x24u, dword_4F07508);
    for ( result = (unsigned int)word_4F06418[0] - 9;
          (unsigned __int16)(word_4F06418[0] - 9) > 1u;
          result = (unsigned int)word_4F06418[0] - 9 )
    {
      sub_7B8B50(0x24u, dword_4F07508, v12, v13, v14, v15);
    }
  }
  else if ( *(_DWORD *)(qword_4F5FCD0 + 12 * qword_4D03CD8 + 8) )
  {
    sub_6851C0(0x26u, dword_4F07508);
    for ( result = (unsigned int)word_4F06418[0] - 9;
          (unsigned __int16)(word_4F06418[0] - 9) > 1u;
          result = (unsigned int)word_4F06418[0] - 9 )
    {
      sub_7B8B50(0x26u, dword_4F07508, v2, v3, v4, v5);
    }
  }
  else
  {
    v7 = a1;
    result = sub_7AFE70();
    v11 = qword_4D03CD0 + 1;
    if ( qword_4D03CD0 + 1 == qword_4D03CD8 )
    {
      result = (unsigned int)(result - 2);
      if ( (result & 0xFD) != 0 )
      {
        a1 = 2;
        result = (__int64)sub_7AFEC0(2);
      }
    }
    if ( v7 )
    {
      while ( (unsigned __int16)(word_4F06418[0] - 9) > 1u )
        sub_7B8B50(a1, a2, v11, v8, v9, v10);
      return sub_856950(0, (__int64)a2, v11, v8, v9, v10);
    }
  }
  return result;
}

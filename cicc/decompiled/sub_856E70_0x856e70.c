// Function: sub_856E70
// Address: 0x856e70
//
__int64 __fastcall sub_856E70(unsigned __int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rsi
  unsigned __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 result; // rax
  int v9; // r13d
  char v10; // dl
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9

  if ( qword_4D03CD8 <= qword_4D03CD0 )
  {
    sub_6851C0(0x24u, dword_4F07508);
    for ( result = (unsigned int)word_4F06418[0] - 9;
          (unsigned __int16)(word_4F06418[0] - 9) > 1u;
          result = (unsigned int)word_4F06418[0] - 9 )
    {
      sub_7B8B50(0x24u, dword_4F07508, v20, v21, v22, v23);
    }
  }
  else
  {
    v1 = *(unsigned int *)(qword_4F5FCD0 + 12 * qword_4D03CD8 + 8);
    if ( (_DWORD)v1 )
    {
      if ( dword_4F077C4 == 1 )
      {
        v2 = (__int64)dword_4F07508;
        v3 = 38;
        sub_684B30(0x26u, dword_4F07508);
      }
      else
      {
        v2 = 38;
        v3 = 7;
        sub_684AC0(7u, 0x26u);
      }
      for ( result = (unsigned int)word_4F06418[0] - 9;
            (unsigned __int16)(word_4F06418[0] - 9) > 1u;
            result = (unsigned int)word_4F06418[0] - 9 )
      {
        sub_7B8B50(v3, (unsigned int *)v2, v4, v5, v6, v7);
      }
    }
    else
    {
      v9 = a1;
      v10 = sub_7AFE70();
      v13 = qword_4D03CD8;
      v14 = qword_4D03CD0 + 1;
      if ( qword_4D03CD0 + 1 == qword_4D03CD8 && ((v10 - 2) & 0xFD) != 0 )
      {
        a1 = 2;
        sub_7AFEC0(2);
        v13 = qword_4D03CD8;
      }
      v15 = qword_4F5FCD0;
      *(_DWORD *)(qword_4F5FCD0 + 12 * v13 + 8) = 1;
      sub_7B8B50(a1, (unsigned int *)v1, v15, v14, v11, v12);
      result = (__int64)word_4F06418;
      if ( word_4F06418[0] != 10 )
        result = sub_855DA0(a1, v1, v16, v17, v18, v19);
      if ( v9 )
        return sub_856950(0, v1, v16, v17, v18, v19);
    }
  }
  return result;
}

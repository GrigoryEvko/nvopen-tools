// Function: sub_855E20
// Address: 0x855e20
//
__int64 __fastcall sub_855E20(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 result; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  char v17; // al

  v6 = qword_4D03CD8;
  if ( qword_4D03CD8 <= qword_4D03CD0 )
  {
    sub_6851C0(0x24u, dword_4F07508);
    for ( result = (unsigned int)word_4F06418[0] - 9;
          (unsigned __int16)(word_4F06418[0] - 9) > 1u;
          result = (unsigned int)word_4F06418[0] - 9 )
    {
      sub_7B8B50(0x24u, dword_4F07508, v13, v14, v15, v16);
    }
  }
  else
  {
    v7 = qword_4D03CD0 + 1;
    if ( qword_4D03CD8 == qword_4D03CD0 + 1 )
    {
      v17 = sub_7AFE70();
      if ( v17 == 3 )
      {
        a1 = 1;
        sub_7AFEC0(1);
        v6 = qword_4D03CD8;
      }
      else
      {
        if ( v17 != 4 )
        {
          a1 = 2;
          sub_7AFEC0(2);
        }
        v6 = qword_4D03CD8;
      }
    }
    qword_4D03CD8 = v6 - 1;
    sub_7B8B50(a1, a2, v7, a4, a5, a6);
    result = (__int64)word_4F06418;
    if ( word_4F06418[0] != 10 )
      return sub_855DA0(a1, (__int64)a2, v8, v9, v10, v11);
  }
  return result;
}

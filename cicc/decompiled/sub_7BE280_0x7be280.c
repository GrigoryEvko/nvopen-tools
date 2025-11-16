// Function: sub_7BE280
// Address: 0x7be280
//
__int64 __fastcall sub_7BE280(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  unsigned __int16 v6; // bx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9

  v6 = a1;
  if ( word_4F06418[0] != (_WORD)a1 )
  {
    a1 = (unsigned int)a2;
    a2 = (unsigned int)a3;
    ++*(_BYTE *)(qword_4F061C8 + v6 + 8LL);
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    if ( a4 )
    {
      sub_7BE200(a1, a3, a4);
      sub_7BE180(a1, a2, v7, v8, v9, v10);
    }
    else
    {
      sub_6851D0(a1);
    }
    --*(_BYTE *)(qword_4F061C8 + v6 + 8LL);
    if ( word_4F06418[0] != v6 )
      return 0;
  }
  sub_7B8B50(a1, (unsigned int *)a2, a3, (__int64)a4, a5, a6);
  return 1;
}

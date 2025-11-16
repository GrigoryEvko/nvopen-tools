// Function: sub_7BE5B0
// Address: 0x7be5b0
//
_BOOL8 __fastcall sub_7BE5B0(unsigned __int16 a1, unsigned int a2, unsigned int a3, __int64 *a4)
{
  _BOOL8 result; // rax
  __int64 v6; // rdi
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9

  result = 1;
  if ( word_4F06418[0] != a1 )
  {
    v6 = a2;
    v7 = a3;
    ++*(_BYTE *)(qword_4F061C8 + a1 + 8LL);
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    if ( a4 )
    {
      sub_7BE200(v6, a3, a4);
      sub_7BE180(v6, v7, v8, v9, v10, v11);
    }
    else
    {
      sub_6851D0(v6);
    }
    --*(_BYTE *)(qword_4F061C8 + a1 + 8LL);
    return word_4F06418[0] == a1;
  }
  return result;
}

// Function: sub_88DF70
// Address: 0x88df70
//
_QWORD *__fastcall sub_88DF70(__int64 *a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rdi
  _QWORD *result; // rax

  if ( word_4F06418[0] != 163 && word_4F06418[0] != 9 && (unsigned __int16)(word_4F06418[0] - 55) > 1u
    || (v9 = *a1) == 0
    || (result = (_QWORD *)sub_8D2310(v9), !(_DWORD)result) )
  {
    if ( !a3 )
      sub_6851C0(0x332u, &dword_4F063F8);
    result = sub_88DE40(a2, a4);
    *a1 = (__int64)result;
    *(_BYTE *)(a5 + 80) |= 2u;
  }
  return result;
}

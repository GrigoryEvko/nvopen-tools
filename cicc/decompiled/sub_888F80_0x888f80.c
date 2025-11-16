// Function: sub_888F80
// Address: 0x888f80
//
__int64 __fastcall sub_888F80(__int64 a1, char **a2)
{
  _QWORD *v3; // rax
  __int64 result; // rax
  __int64 i; // r13
  char *v6; // r14
  char v7; // si
  _QWORD *v8; // rax

  v3 = sub_72B620(a1, 1);
  sub_888EB0(*a2, (__int64)v3);
  result = (unsigned int)qword_4F077B4;
  if ( (_DWORD)qword_4F077B4 )
  {
    result = (__int64)&qword_4F077A0;
    if ( qword_4F077A0 > 0x1ADAFu )
    {
      for ( i = 2; i != 5; ++i )
      {
        v6 = a2[i - 1];
        v7 = i;
        v8 = sub_72B620(a1, v7);
        result = sub_888EB0(v6, (__int64)v8);
      }
    }
  }
  return result;
}

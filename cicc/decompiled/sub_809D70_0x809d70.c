// Function: sub_809D70
// Address: 0x809d70
//
_QWORD *__fastcall sub_809D70(__int64 a1, unsigned __int8 a2, int a3)
{
  _QWORD *result; // rax
  __int64 v4; // rax
  _QWORD *i; // rbx
  _QWORD *v6; // r12
  _DWORD v7[12]; // [rsp+0h] [rbp-30h] BYREF

  dword_4F18B88 = a3;
  result = (_QWORD *)sub_737670(a1, a2, (__int64 (__fastcall *)(__int64, _QWORD, _DWORD *))sub_80AE00, v7, 15);
  if ( a2 == 7 )
  {
    if ( (*(_BYTE *)(a1 + 170) & 0x10) == 0 )
      return result;
    result = *(_QWORD **)(a1 + 216);
    i = (_QWORD *)*result;
  }
  else
  {
    v4 = *(_QWORD *)(a1 + 152);
    for ( i = *(_QWORD **)(a1 + 240); *(_BYTE *)(v4 + 140) == 12; v4 = *(_QWORD *)(v4 + 160) )
      ;
    result = *(_QWORD **)(v4 + 168);
    v6 = (_QWORD *)*result;
    if ( *result )
    {
      do
      {
        result = (_QWORD *)sub_737670(v6[1], 6u, (__int64 (__fastcall *)(__int64, _QWORD, _DWORD *))sub_80AE00, v7, 15);
        v6 = (_QWORD *)*v6;
      }
      while ( v6 );
    }
  }
  for ( ; i; i = (_QWORD *)*i )
  {
    while ( *((_BYTE *)i + 8) )
    {
      i = (_QWORD *)*i;
      if ( !i )
        return result;
    }
    result = (_QWORD *)sub_737670(i[4], 6u, (__int64 (__fastcall *)(__int64, _QWORD, _DWORD *))sub_80AE00, v7, 15);
  }
  return result;
}

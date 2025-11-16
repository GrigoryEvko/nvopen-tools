// Function: sub_A4AAA0
// Address: 0xa4aaa0
//
_BYTE *__fastcall sub_A4AAA0(_QWORD *a1, char *a2, char *a3)
{
  char *v4; // rbx
  _BYTE *result; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = a2;
  if ( !a2 && a3 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v9[0] = (a3 - a2) >> 3;
  if ( (unsigned __int64)(a3 - a2) > 0x78 )
  {
    result = (_BYTE *)sub_22409D0(a1, v9, 0);
    v8 = v9[0];
    *a1 = result;
    a1[2] = v8;
  }
  else
  {
    result = (_BYTE *)*a1;
  }
  if ( a2 != a3 )
  {
    do
    {
      v6 = *(_QWORD *)v4;
      v4 += 8;
      *result++ = v6;
    }
    while ( a3 != v4 );
    result = (_BYTE *)*a1;
  }
  v7 = v9[0];
  a1[1] = v9[0];
  result[v7] = 0;
  return result;
}

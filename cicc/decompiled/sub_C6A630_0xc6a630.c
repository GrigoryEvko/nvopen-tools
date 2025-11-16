// Function: sub_C6A630
// Address: 0xc6a630
//
__int64 __fastcall sub_C6A630(char *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rbp
  char *v4; // rsi
  char *v5; // rax
  __int64 result; // rax
  _QWORD v8[4]; // [rsp-20h] [rbp-20h] BYREF

  v4 = &a1[a2];
  if ( v4 == a1 )
    return 1;
  v8[3] = v3;
  v5 = a1;
  while ( *v5 >= 0 )
  {
    if ( v4 == ++v5 )
      return 1;
  }
  v8[0] = a1;
  if ( (unsigned __int8)sub_F037A0(v8) )
    return 1;
  result = 0;
  if ( a3 )
    *a3 = v8[0] - (_QWORD)a1;
  return result;
}

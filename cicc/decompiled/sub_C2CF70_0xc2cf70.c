// Function: sub_C2CF70
// Address: 0xc2cf70
//
__int64 __fastcall sub_C2CF70(_QWORD *a1)
{
  __int64 (*v1)(void); // rax
  char v2; // al
  unsigned __int64 v3; // rsi
  __int64 result; // rax

  v1 = *(__int64 (**)(void))(*a1 + 32LL);
  if ( (char *)v1 == (char *)sub_C2CF50 )
  {
    if ( !a1[24] )
    {
LABEL_5:
      v3 = a1[26];
      if ( a1[27] <= v3 )
        goto LABEL_12;
      while ( 1 )
      {
        result = sub_C29DF0((__int64)a1, v3);
        if ( (_DWORD)result )
          return result;
        v3 = a1[26];
        if ( v3 >= a1[27] )
          goto LABEL_12;
      }
    }
    v2 = sub_C2CAB0((__int64)a1);
  }
  else
  {
    v2 = v1();
  }
  if ( !v2 )
    goto LABEL_5;
  result = sub_C2BD40((__int64)a1, (__int64)(a1 + 61), a1 + 1);
  if ( !(_DWORD)result )
  {
    a1[26] = a1[27];
LABEL_12:
    sub_C1AFD0();
    return 0;
  }
  return result;
}

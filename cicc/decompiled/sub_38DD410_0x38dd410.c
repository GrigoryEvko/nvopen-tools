// Function: sub_38DD410
// Address: 0x38dd410
//
_QWORD *__fastcall sub_38DD410(__int64 a1, __int64 a2, char a3, char a4, unsigned __int64 a5)
{
  _QWORD *result; // rax
  __int64 v9; // rdi
  const char *v10; // rax
  const char *v11; // [rsp+0h] [rbp-50h] BYREF
  char v12; // [rsp+10h] [rbp-40h]
  char v13; // [rsp+11h] [rbp-3Fh]

  result = (_QWORD *)sub_38DD280(a1, a5);
  if ( result )
  {
    if ( result[8] )
    {
      v13 = 1;
      v9 = *(_QWORD *)(a1 + 8);
      v10 = "Chained unwind areas can't have handlers!";
    }
    else
    {
      result[2] = a2;
      if ( a4 )
      {
        if ( !a3 )
        {
LABEL_6:
          *((_BYTE *)result + 57) = 1;
          return result;
        }
LABEL_5:
        *((_BYTE *)result + 56) = 1;
        if ( !a4 )
          return result;
        goto LABEL_6;
      }
      if ( a3 )
        goto LABEL_5;
      v13 = 1;
      v9 = *(_QWORD *)(a1 + 8);
      v10 = "Don't know what kind of handler this is!";
    }
    v11 = v10;
    v12 = 3;
    return sub_38BE3D0(v9, a5, (__int64)&v11);
  }
  return result;
}

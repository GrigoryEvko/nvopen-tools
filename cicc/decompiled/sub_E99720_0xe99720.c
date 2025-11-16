// Function: sub_E99720
// Address: 0xe99720
//
__int64 __fastcall sub_E99720(__int64 a1, __int64 a2, char a3, char a4, _QWORD *a5)
{
  __int64 result; // rax
  __int64 v9; // rdi
  const char *v10; // rax
  const char *v11; // [rsp+0h] [rbp-60h] BYREF
  char v12; // [rsp+20h] [rbp-40h]
  char v13; // [rsp+21h] [rbp-3Fh]

  result = sub_E99590(a1, a5);
  if ( result )
  {
    if ( *(_QWORD *)(result + 80) )
    {
      v13 = 1;
      v9 = *(_QWORD *)(a1 + 8);
      v10 = "Chained unwind areas can't have handlers!";
    }
    else
    {
      *(_QWORD *)(result + 24) = a2;
      if ( a4 )
      {
        if ( !a3 )
        {
LABEL_6:
          *(_BYTE *)(result + 73) = 1;
          return result;
        }
LABEL_5:
        *(_BYTE *)(result + 72) = 1;
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
    return sub_E66880(v9, a5, (__int64)&v11);
  }
  return result;
}

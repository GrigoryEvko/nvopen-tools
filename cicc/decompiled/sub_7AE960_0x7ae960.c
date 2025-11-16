// Function: sub_7AE960
// Address: 0x7ae960
//
__int64 __fastcall sub_7AE960(__int64 a1, __int64 a2, int a3, int a4, int a5, int a6)
{
  __int64 result; // rax
  _QWORD *v10; // r11
  __int64 v11; // r10
  _QWORD *v12; // rcx
  __int64 v13; // rsi

  result = *(_QWORD *)(a1 + 8);
  if ( result )
  {
    v10 = 0;
    v11 = 0;
    v12 = 0;
    v13 = 0;
    while ( *(_DWORD *)(result + 28) != a3 )
    {
      if ( *(_BYTE *)(result + 26) != 3 )
      {
        v10 = v12;
        v11 = v13;
        v12 = (_QWORD *)result;
        v13 = *(_QWORD *)result;
      }
      if ( !*(_QWORD *)result )
        goto LABEL_16;
      result = *(_QWORD *)result;
    }
LABEL_8:
    if ( a4 )
    {
      v13 = v11;
      v12 = v10;
    }
    *(_QWORD *)(a2 + 8) = v13;
    *(_QWORD *)(a2 + 16) = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 16) = v12;
    *v12 = 0;
    result = sub_7AE210(a1);
    goto LABEL_11;
  }
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
LABEL_16:
  if ( !a5 )
    goto LABEL_8;
LABEL_11:
  if ( a6 )
  {
    result = qword_4F08538;
    if ( qword_4F08538 )
    {
      if ( *(_QWORD *)(qword_4F08538 + 24) == a1 )
        *(_QWORD *)(qword_4F08538 + 24) = a2;
    }
  }
  return result;
}

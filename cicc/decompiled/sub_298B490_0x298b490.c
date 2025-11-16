// Function: sub_298B490
// Address: 0x298b490
//
unsigned __int64 __fastcall sub_298B490(_BYTE *a1, char a2)
{
  _QWORD *v3; // r13
  char *v4; // r12
  unsigned __int64 v5; // rsi
  char *v6; // rax
  char *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 result; // rax
  char *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx

  *a1 = a2;
  v3 = sub_C52410();
  v4 = (char *)(v3 + 1);
  v5 = sub_C959E0();
  v6 = (char *)v3[2];
  if ( v6 )
  {
    v7 = (char *)(v3 + 1);
    do
    {
      while ( 1 )
      {
        v8 = *((_QWORD *)v6 + 2);
        v9 = *((_QWORD *)v6 + 3);
        if ( v5 <= *((_QWORD *)v6 + 4) )
          break;
        v6 = (char *)*((_QWORD *)v6 + 3);
        if ( !v9 )
          goto LABEL_6;
      }
      v7 = v6;
      v6 = (char *)*((_QWORD *)v6 + 2);
    }
    while ( v8 );
LABEL_6:
    if ( v4 != v7 && v5 >= *((_QWORD *)v7 + 4) )
      v4 = v7;
  }
  result = (unsigned __int64)sub_C52410() + 8;
  if ( v4 != (char *)result )
  {
    result = *((_QWORD *)v4 + 7);
    if ( result )
    {
      v11 = v4 + 48;
      do
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(result + 16);
          v13 = *(_QWORD *)(result + 24);
          if ( *(_DWORD *)(result + 32) >= dword_5007888 )
            break;
          result = *(_QWORD *)(result + 24);
          if ( !v13 )
            goto LABEL_15;
        }
        v11 = (char *)result;
        result = *(_QWORD *)(result + 16);
      }
      while ( v12 );
LABEL_15:
      if ( v4 + 48 != v11 && dword_5007888 >= *((_DWORD *)v11 + 8) )
      {
        result = *((unsigned int *)v11 + 9);
        if ( (_DWORD)result )
        {
          result = (unsigned __int8)qword_5007908;
          *a1 = qword_5007908;
        }
      }
    }
  }
  return result;
}

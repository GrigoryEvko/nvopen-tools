// Function: sub_2FE6BE0
// Address: 0x2fe6be0
//
char *__fastcall sub_2FE6BE0(__int64 a1, char a2)
{
  _QWORD *v4; // r14
  char *v5; // r13
  unsigned __int64 v6; // rsi
  char *v7; // rax
  char *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  char *result; // rax
  char *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx

  v4 = sub_C52410();
  v5 = (char *)(v4 + 1);
  v6 = sub_C959E0();
  v7 = (char *)v4[2];
  if ( v7 )
  {
    v8 = (char *)(v4 + 1);
    do
    {
      while ( 1 )
      {
        v9 = *((_QWORD *)v7 + 2);
        v10 = *((_QWORD *)v7 + 3);
        if ( v6 <= *((_QWORD *)v7 + 4) )
          break;
        v7 = (char *)*((_QWORD *)v7 + 3);
        if ( !v10 )
          goto LABEL_6;
      }
      v8 = v7;
      v7 = (char *)*((_QWORD *)v7 + 2);
    }
    while ( v9 );
LABEL_6:
    if ( v5 != v8 && v6 >= *((_QWORD *)v8 + 4) )
      v5 = v8;
  }
  result = (char *)sub_C52410() + 8;
  if ( v5 == result )
    goto LABEL_18;
  result = (char *)*((_QWORD *)v5 + 7);
  if ( !result )
    goto LABEL_18;
  v12 = v5 + 48;
  do
  {
    while ( 1 )
    {
      v13 = *((_QWORD *)result + 2);
      v14 = *((_QWORD *)result + 3);
      if ( *((_DWORD *)result + 8) >= dword_5026DC8 )
        break;
      result = (char *)*((_QWORD *)result + 3);
      if ( !v14 )
        goto LABEL_15;
    }
    v12 = result;
    result = (char *)*((_QWORD *)result + 2);
  }
  while ( v13 );
LABEL_15:
  if ( v5 + 48 == v12
    || dword_5026DC8 < *((_DWORD *)v12 + 8)
    || (result = (char *)*((unsigned int *)v12 + 9), !(_DWORD)result) )
  {
LABEL_18:
    *(_BYTE *)(a1 + 56) = a2;
  }
  return result;
}

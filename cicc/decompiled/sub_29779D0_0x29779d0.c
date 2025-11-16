// Function: sub_29779D0
// Address: 0x29779d0
//
unsigned __int64 __fastcall sub_29779D0(_BYTE *a1)
{
  _QWORD *v2; // r13
  char *v3; // r12
  unsigned __int64 v4; // rsi
  char *v5; // rax
  char *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned __int64 result; // rax
  char *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx

  v2 = sub_C52410();
  v3 = (char *)(v2 + 1);
  v4 = sub_C959E0();
  v5 = (char *)v2[2];
  if ( v5 )
  {
    v6 = (char *)(v2 + 1);
    do
    {
      while ( 1 )
      {
        v7 = *((_QWORD *)v5 + 2);
        v8 = *((_QWORD *)v5 + 3);
        if ( v4 <= *((_QWORD *)v5 + 4) )
          break;
        v5 = (char *)*((_QWORD *)v5 + 3);
        if ( !v8 )
          goto LABEL_6;
      }
      v6 = v5;
      v5 = (char *)*((_QWORD *)v5 + 2);
    }
    while ( v7 );
LABEL_6:
    if ( v3 != v6 && v4 >= *((_QWORD *)v6 + 4) )
      v3 = v6;
  }
  result = (unsigned __int64)sub_C52410() + 8;
  if ( v3 != (char *)result )
  {
    result = *((_QWORD *)v3 + 7);
    if ( result )
    {
      v10 = v3 + 48;
      do
      {
        while ( 1 )
        {
          v11 = *(_QWORD *)(result + 16);
          v12 = *(_QWORD *)(result + 24);
          if ( *(_DWORD *)(result + 32) >= dword_5006E08 )
            break;
          result = *(_QWORD *)(result + 24);
          if ( !v12 )
            goto LABEL_15;
        }
        v10 = (char *)result;
        result = *(_QWORD *)(result + 16);
      }
      while ( v11 );
LABEL_15:
      if ( v3 + 48 != v10 && dword_5006E08 >= *((_DWORD *)v10 + 8) )
      {
        result = *((unsigned int *)v10 + 9);
        if ( (_DWORD)result )
        {
          result = (unsigned __int8)qword_5006E88;
          *a1 = qword_5006E88;
        }
      }
    }
  }
  return result;
}

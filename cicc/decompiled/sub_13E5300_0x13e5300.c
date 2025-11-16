// Function: sub_13E5300
// Address: 0x13e5300
//
char *__fastcall sub_13E5300(__int64 a1, char **a2)
{
  char *result; // rax
  char *v3; // r14
  char *v4; // rbx
  char *v6; // r12
  unsigned __int64 v7; // rsi
  char *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rdi
  _BYTE *v12; // rsi
  char *v13; // [rsp+8h] [rbp-28h] BYREF

  result = *a2;
  v3 = a2[5];
  v4 = a2[4];
  v13 = *a2;
  if ( v4 != v3 )
  {
    v6 = (char *)(a1 + 168);
    do
    {
      result = *(char **)(a1 + 176);
      v7 = *(_QWORD *)v4;
      if ( !result )
        goto LABEL_17;
      v8 = v6;
      do
      {
        while ( 1 )
        {
          v9 = *((_QWORD *)result + 2);
          v10 = *((_QWORD *)result + 3);
          if ( *((_QWORD *)result + 4) >= v7 )
            break;
          result = (char *)*((_QWORD *)result + 3);
          if ( !v10 )
            goto LABEL_8;
        }
        v8 = result;
        result = (char *)*((_QWORD *)result + 2);
      }
      while ( v9 );
LABEL_8:
      if ( v6 == v8 || *((_QWORD *)v8 + 4) > v7 )
LABEL_17:
        BUG();
      v11 = *((_QWORD *)v8 + 5);
      v12 = *(_BYTE **)(v11 + 64);
      if ( v12 == *(_BYTE **)(v11 + 72) )
      {
        result = sub_1292090(v11 + 56, v12, &v13);
      }
      else
      {
        if ( v12 )
        {
          result = v13;
          *(_QWORD *)v12 = v13;
          v12 = *(_BYTE **)(v11 + 64);
        }
        *(_QWORD *)(v11 + 64) = v12 + 8;
      }
      v4 += 8;
    }
    while ( v3 != v4 );
  }
  return result;
}

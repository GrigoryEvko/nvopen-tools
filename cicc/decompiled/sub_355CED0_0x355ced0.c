// Function: sub_355CED0
// Address: 0x355ced0
//
_QWORD *__fastcall sub_355CED0(_QWORD *a1, char *a2, __int64 a3, char **a4)
{
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v7; // rax
  char *v8; // rdi
  char *v9; // r13
  char *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // r15
  __int64 v14; // rax
  bool v15; // sf
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  char *v19; // rax
  char *v20; // rsi
  char *v21; // rdx
  char *v22; // rax

  v4 = a3 - (_QWORD)a2;
  v5 = v4 >> 3;
  v7 = (__int64)a4[2];
  v8 = *a4;
  if ( v4 > 0 )
  {
    v9 = a2;
    do
    {
      while ( 1 )
      {
        v10 = v9;
        v11 = (v7 - (__int64)v8) >> 3;
        if ( v11 > v5 )
          v11 = v5;
        v12 = 8 * v11;
        v13 = v11;
        v9 += 8 * v11;
        if ( v9 != v10 )
        {
          memmove(v8, v10, 8 * v11);
          v8 = *a4;
        }
        v14 = (v8 - a4[1]) >> 3;
        v15 = v13 + v14 < 0;
        v16 = v13 + v14;
        v17 = v16;
        if ( !v15 )
          break;
        v18 = ~((unsigned __int64)~v16 >> 6);
LABEL_11:
        v5 -= v13;
        v19 = &a4[3][8 * v18];
        a4[3] = v19;
        v20 = *(char **)v19;
        v7 = *(_QWORD *)v19 + 512LL;
        v8 = &v20[8 * (v17 - (v18 << 6))];
        a4[1] = v20;
        a4[2] = (char *)v7;
        *a4 = v8;
        if ( v5 <= 0 )
          goto LABEL_12;
      }
      if ( v16 > 63 )
      {
        v18 = v16 >> 6;
        goto LABEL_11;
      }
      v8 += v12;
      v5 -= v13;
      v7 = (__int64)a4[2];
      *a4 = v8;
    }
    while ( v5 > 0 );
  }
LABEL_12:
  v21 = a4[1];
  a1[2] = v7;
  v22 = a4[3];
  *a1 = v8;
  a1[3] = v22;
  a1[1] = v21;
  return a1;
}

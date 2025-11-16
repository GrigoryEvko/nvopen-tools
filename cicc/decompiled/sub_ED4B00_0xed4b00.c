// Function: sub_ED4B00
// Address: 0xed4b00
//
signed __int64 __fastcall sub_ED4B00(char *a1, char *a2, __int64 a3)
{
  signed __int64 result; // rax
  unsigned __int64 v5; // r13
  __int64 v6; // rbx
  char *v7; // r10
  _QWORD *v8; // r12
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // r8
  char *v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 *v16; // rdi
  unsigned __int64 v17; // r15
  _QWORD *v18; // rcx
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 i; // rsi
  unsigned __int64 *v29; // r13
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // r8
  __int64 v32; // rbx
  unsigned __int64 *v33; // [rsp-40h] [rbp-40h]

  result = a2 - a1;
  if ( a2 - a1 <= 256 )
    return result;
  v5 = (unsigned __int64)a2;
  v6 = a3;
  if ( !a3 )
    goto LABEL_24;
  v7 = a2;
  v8 = a1 + 16;
  v33 = (unsigned __int64 *)(a1 + 32);
  while ( 2 )
  {
    v9 = *((_QWORD *)a1 + 2);
    v10 = *((_QWORD *)v7 - 2);
    --v6;
    v11 = *(_QWORD *)a1;
    v12 = &a1[16 * ((__int64)(((v7 - a1) >> 4) + ((unsigned __int64)(v7 - a1) >> 63)) >> 1)];
    v13 = *(_QWORD *)v12;
    if ( v9 >= *(_QWORD *)v12 )
    {
      if ( v10 > v9 )
        goto LABEL_7;
      if ( v10 > v13 )
      {
LABEL_18:
        *(_QWORD *)a1 = v10;
        v23 = *((_QWORD *)v7 - 1);
        *((_QWORD *)v7 - 2) = v11;
        v24 = *((_QWORD *)a1 + 1);
        *((_QWORD *)a1 + 1) = v23;
        *((_QWORD *)v7 - 1) = v24;
        v11 = *((_QWORD *)a1 + 2);
        v9 = *(_QWORD *)a1;
        goto LABEL_8;
      }
LABEL_23:
      *(_QWORD *)a1 = v13;
      v25 = *((_QWORD *)v12 + 1);
      *(_QWORD *)v12 = v11;
      v26 = *((_QWORD *)a1 + 1);
      *((_QWORD *)a1 + 1) = v25;
      *((_QWORD *)v12 + 1) = v26;
      v11 = *((_QWORD *)a1 + 2);
      v9 = *(_QWORD *)a1;
      goto LABEL_8;
    }
    if ( v10 > v13 )
      goto LABEL_23;
    if ( v10 > v9 )
      goto LABEL_18;
LABEL_7:
    v14 = *((_QWORD *)a1 + 1);
    v15 = *((_QWORD *)a1 + 3);
    *(_QWORD *)a1 = v9;
    *((_QWORD *)a1 + 2) = v11;
    *((_QWORD *)a1 + 1) = v15;
    *((_QWORD *)a1 + 3) = v14;
LABEL_8:
    v16 = v33;
    v17 = (unsigned __int64)v8;
    v18 = v7;
    while ( 1 )
    {
      v5 = v17;
      if ( v9 > v11 )
        goto LABEL_15;
      v19 = *(v18 - 2);
      if ( v19 <= v9 )
      {
        v18 -= 2;
      }
      else
      {
        v20 = v18 - 4;
        do
        {
          v18 = v20;
          v19 = *v20;
          v20 -= 2;
        }
        while ( v19 > v9 );
      }
      if ( v17 >= (unsigned __int64)v18 )
        break;
      *(v16 - 2) = v19;
      v21 = v18[1];
      *v18 = v11;
      v22 = *(v16 - 1);
      *(v16 - 1) = v21;
      v18[1] = v22;
      v9 = *(_QWORD *)a1;
LABEL_15:
      v11 = *v16;
      v17 += 16LL;
      v16 += 2;
    }
    sub_ED4B00(v17, v7, v6);
    result = v17 - (_QWORD)a1;
    if ( (__int64)(v17 - (_QWORD)a1) > 256 )
    {
      if ( v6 )
      {
        v7 = (char *)v17;
        continue;
      }
LABEL_24:
      v27 = result >> 4;
      for ( i = ((result >> 4) - 2) >> 1; ; --i )
      {
        sub_ECFE10((__int64)a1, i, v27, *(_QWORD *)&a1[16 * i], *(_QWORD *)&a1[16 * i + 8]);
        if ( !i )
          break;
      }
      v29 = (unsigned __int64 *)(v5 - 16);
      do
      {
        v30 = *v29;
        v31 = v29[1];
        v32 = (char *)v29 - a1;
        v29 -= 2;
        v29[2] = *(_QWORD *)a1;
        v29[3] = *((_QWORD *)a1 + 1);
        result = (signed __int64)sub_ECFE10((__int64)a1, 0, v32 >> 4, v30, v31);
      }
      while ( v32 > 16 );
    }
    return result;
  }
}

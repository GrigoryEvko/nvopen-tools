// Function: sub_E122B0
// Address: 0xe122b0
//
__int64 __fastcall sub_E122B0(__int64 a1, __int64 *a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  char *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r12
  char *v13; // r13
  char *v14; // r13
  unsigned __int64 v15; // rdx
  char *v16; // rdi
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  char *v20; // rdi
  __int64 result; // rax
  unsigned __int64 v22; // rdx
  char *v23; // rdi
  char *v24; // r13
  unsigned int v25; // edi
  __int64 v26; // rcx

  v4 = (char *)a2[1];
  v5 = a2[2];
  v6 = (char *)*a2;
  if ( (unsigned __int64)(v4 + 5) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 997);
    v8 = 2 * v5;
    if ( v7 > v8 )
      a2[2] = v7;
    else
      a2[2] = v8;
    v9 = realloc(v6);
    *a2 = v9;
    v6 = (char *)v9;
    if ( !v9 )
LABEL_28:
      abort();
    v4 = (char *)a2[1];
  }
  v10 = &v6[(_QWORD)v4];
  *(_DWORD *)v10 = 979661939;
  v10[4] = 58;
  v11 = a2[1] + 5;
  a2[1] = v11;
  switch ( *(_DWORD *)(a1 + 12) )
  {
    case 0:
      v14 = "allocator";
      v12 = 9;
      goto LABEL_9;
    case 1:
      v14 = "basic_string";
      v12 = 12;
      goto LABEL_9;
    case 2:
      v12 = 6;
      v13 = "basic_string";
      goto LABEL_8;
    case 3:
      v12 = 7;
      v13 = "basic_istream";
      goto LABEL_8;
    case 4:
      v12 = 7;
      v13 = "basic_ostream";
      goto LABEL_8;
    case 5:
      v12 = 8;
      v13 = "basic_iostream";
LABEL_8:
      v14 = v13 + 6;
LABEL_9:
      v15 = a2[2];
      v16 = (char *)*a2;
      if ( v12 + v11 <= v15 )
        goto LABEL_14;
      v17 = v12 + v11 + 992;
      v18 = 2 * v15;
      if ( v17 > v18 )
        a2[2] = v17;
      else
        a2[2] = v18;
      v19 = realloc(v16);
      *a2 = v19;
      v16 = (char *)v19;
      if ( !v19 )
        goto LABEL_28;
      v11 = a2[1];
LABEL_14:
      v20 = &v16[v11];
      result = (unsigned int)v12;
      if ( (unsigned int)v12 < 8 )
      {
        *(_DWORD *)v20 = *(_DWORD *)v14;
        *(_DWORD *)&v20[(unsigned int)v12 - 4] = *(_DWORD *)&v14[(unsigned int)v12 - 4];
LABEL_16:
        a2[1] += v12;
        return result;
      }
      v22 = (unsigned __int64)(v20 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v20 = *(_QWORD *)v14;
      *(_QWORD *)&v20[v12 - 8] = *(_QWORD *)&v14[v12 - 8];
      v23 = &v20[-v22];
      v24 = (char *)(v14 - v23);
      result = ((_DWORD)v12 + (_DWORD)v23) & 0xFFFFFFF8;
      if ( (unsigned int)result < 8 )
        goto LABEL_16;
      v25 = (v12 + (_DWORD)v23) & 0xFFFFFFF8;
      LODWORD(result) = 0;
      do
      {
        v26 = (unsigned int)result;
        result = (unsigned int)(result + 8);
        *(_QWORD *)(v22 + v26) = *(_QWORD *)&v24[v26];
      }
      while ( (unsigned int)result < v25 );
      a2[1] += v12;
      return result;
  }
}

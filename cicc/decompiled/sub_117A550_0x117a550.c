// Function: sub_117A550
// Address: 0x117a550
//
char __fastcall sub_117A550(__int64 a1, __int64 a2, int a3)
{
  int *v5; // rax
  int v6; // eax
  __int64 v7; // rbx
  char *v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned int v11; // r15d
  char *v12; // rbx
  __int64 v13; // rdi
  __int64 v14; // rsi
  int v15; // edx
  __int64 v16; // rdi
  __int64 v17; // rsi
  int v18; // edx
  __int64 v19; // rdi
  __int64 v20; // rsi
  int v21; // edx
  __int64 v22; // rsi
  int v23; // edx
  char result; // al
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  int *v27; // rax
  int v28; // eax
  int *v29; // rax
  int v30; // r14d
  __int64 v31; // rsi
  int v32; // edx
  __int64 v33; // rsi
  int v34; // edx
  __int64 v35; // rsi
  int v36; // edx
  char *v37; // [rsp+8h] [rbp-38h]

  v5 = (int *)sub_C94E20((__int64)qword_4F862D0);
  if ( v5 )
    v6 = *v5;
  else
    v6 = qword_4F862D0[2];
  if ( v6 == a3 )
    return 0;
  if ( a3 )
  {
    if ( *(_BYTE *)(a2 + 28) )
    {
      v25 = *(_QWORD **)(a2 + 8);
      v26 = &v25[*(unsigned int *)(a2 + 20)];
      if ( v25 == v26 )
        goto LABEL_5;
      while ( a1 != *v25 )
      {
        if ( v26 == ++v25 )
          goto LABEL_5;
      }
    }
    else if ( !sub_C8CA60(a2, a1) )
    {
      goto LABEL_5;
    }
    return 1;
  }
LABEL_5:
  if ( *(_BYTE *)a1 <= 0x1Cu )
    return 0;
  if ( *(_BYTE *)a1 == 84 )
  {
    v27 = (int *)sub_C94E20((__int64)qword_4F862D0);
    if ( v27 )
      v28 = *v27;
    else
      v28 = qword_4F862D0[2];
    if ( v28 - 1 == a3 )
      return 0;
    v29 = (int *)sub_C94E20((__int64)qword_4F862D0);
    if ( v29 )
      v30 = *v29;
    else
      v30 = qword_4F862D0[2];
    a3 = v30 - 2;
  }
  v7 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v8 = *(char **)(a1 - 8);
    v37 = &v8[v7];
  }
  else
  {
    v37 = (char *)a1;
    v8 = (char *)(a1 - v7);
  }
  v9 = v7 >> 5;
  v10 = v7 >> 7;
  if ( v10 )
  {
    v11 = a3 + 1;
    v12 = &v8[128 * v10];
    while ( 1 )
    {
      v22 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
      v23 = *(unsigned __int8 *)(v22 + 8);
      if ( (unsigned int)(v23 - 17) <= 1 )
        LOBYTE(v23) = *(_BYTE *)(**(_QWORD **)(v22 + 16) + 8LL);
      if ( (_BYTE)v23 == 12 && (unsigned __int8)sub_117A550(*(_QWORD *)v8, a2, v11) )
        return v37 != v8;
      v13 = *((_QWORD *)v8 + 4);
      v14 = *(_QWORD *)(v13 + 8);
      v15 = *(unsigned __int8 *)(v14 + 8);
      if ( (unsigned int)(v15 - 17) <= 1 )
        LOBYTE(v15) = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
      if ( (_BYTE)v15 == 12 && (unsigned __int8)sub_117A550(v13, a2, v11) )
        return v37 != v8 + 32;
      v16 = *((_QWORD *)v8 + 8);
      v17 = *(_QWORD *)(v16 + 8);
      v18 = *(unsigned __int8 *)(v17 + 8);
      if ( (unsigned int)(v18 - 17) <= 1 )
        LOBYTE(v18) = *(_BYTE *)(**(_QWORD **)(v17 + 16) + 8LL);
      if ( (_BYTE)v18 == 12 && (unsigned __int8)sub_117A550(v16, a2, v11) )
        return v37 != v8 + 64;
      v19 = *((_QWORD *)v8 + 12);
      v20 = *(_QWORD *)(v19 + 8);
      v21 = *(unsigned __int8 *)(v20 + 8);
      if ( (unsigned int)(v21 - 17) <= 1 )
        LOBYTE(v21) = *(_BYTE *)(**(_QWORD **)(v20 + 16) + 8LL);
      if ( (_BYTE)v21 == 12 && (unsigned __int8)sub_117A550(v19, a2, v11) )
        return v37 != v8 + 96;
      v8 += 128;
      if ( v12 == v8 )
      {
        v9 = (v37 - v8) >> 5;
        break;
      }
    }
  }
  if ( v9 != 2 )
  {
    if ( v9 != 3 )
    {
      if ( v9 != 1 )
        return 0;
      goto LABEL_63;
    }
    v31 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
    v32 = *(unsigned __int8 *)(v31 + 8);
    if ( (unsigned int)(v32 - 17) <= 1 )
      LOBYTE(v32) = *(_BYTE *)(**(_QWORD **)(v31 + 16) + 8LL);
    if ( (_BYTE)v32 == 12 && (unsigned __int8)sub_117A550(*(_QWORD *)v8, a2, (unsigned int)(a3 + 1)) )
      return v37 != v8;
    v8 += 32;
  }
  v33 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
  v34 = *(unsigned __int8 *)(v33 + 8);
  if ( (unsigned int)(v34 - 17) <= 1 )
    LOBYTE(v34) = *(_BYTE *)(**(_QWORD **)(v33 + 16) + 8LL);
  if ( (_BYTE)v34 == 12 && (unsigned __int8)sub_117A550(*(_QWORD *)v8, a2, (unsigned int)(a3 + 1)) )
    return v8 != v37;
  v8 += 32;
LABEL_63:
  v35 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
  v36 = *(unsigned __int8 *)(v35 + 8);
  if ( (unsigned int)(v36 - 17) <= 1 )
    LOBYTE(v36) = *(_BYTE *)(**(_QWORD **)(v35 + 16) + 8LL);
  result = 0;
  if ( (_BYTE)v36 == 12 )
  {
    result = sub_117A550(*(_QWORD *)v8, a2, (unsigned int)(a3 + 1));
    if ( result )
      return v8 != v37;
  }
  return result;
}

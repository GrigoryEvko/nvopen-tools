// Function: sub_153DF40
// Address: 0x153df40
//
unsigned int *__fastcall sub_153DF40(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r9
  unsigned int v7; // r10d
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned int *v10; // rcx
  unsigned int *result; // rax
  unsigned int v12; // ebx
  unsigned int v13; // r14d
  unsigned int v14; // r13d
  unsigned int v15; // r12d
  _BYTE *v16; // r11
  _BYTE *v17; // r11
  unsigned int v18; // r9d
  __int64 v19; // rcx
  __int64 v20; // r11
  __int64 v21; // rsi
  unsigned int v22; // r10d
  unsigned int v23; // r15d
  unsigned int v24; // r14d
  _BYTE *v25; // r9
  unsigned int v26; // r9d
  _BYTE *v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned int v31; // [rsp+8h] [rbp-50h]
  unsigned int v32; // [rsp+Ch] [rbp-4Ch]
  __int64 v33; // [rsp+10h] [rbp-48h]
  unsigned int v34; // [rsp+18h] [rbp-40h]
  __int64 v35; // [rsp+20h] [rbp-38h]
  unsigned __int64 v36; // [rsp+28h] [rbp-30h]

  v5 = (a3 - 1) / 2;
  v35 = a2;
  v34 = a4;
  v33 = a3 & 1;
  v36 = HIDWORD(a4);
  v31 = HIDWORD(a4);
  v32 = a4;
  if ( a2 >= v5 )
  {
    v8 = a2;
    result = (unsigned int *)(a1 + 8 * a2);
    if ( v33 )
      goto LABEL_36;
    goto LABEL_33;
  }
  while ( 1 )
  {
    v6 = *(_QWORD *)(a5 + 208);
    v7 = 0;
    v8 = 2 * (a2 + 1);
    v9 = 16 * (a2 + 1);
    v10 = (unsigned int *)(a1 + v9 - 8);
    result = (unsigned int *)(a1 + v9);
    v12 = v10[1];
    v13 = *v10;
    v14 = *result;
    v15 = result[1];
    v16 = *(_BYTE **)(v6 + 8LL * (v12 - 1));
    if ( *v16 )
    {
      v7 = 1;
      if ( (unsigned __int8)(*v16 - 4) <= 0x1Eu )
        v7 = (v16[1] != 1) + 2;
    }
    v17 = *(_BYTE **)(v6 + 8LL * (v15 - 1));
    v18 = 0;
    if ( *v17 )
    {
      v18 = 1;
      if ( (unsigned __int8)(*v17 - 4) <= 0x1Eu )
        v18 = (v17[1] != 1) + 2;
    }
    if ( v13 > v14 || v13 == v14 && (v7 > v18 || v7 == v18 && v15 < v12) )
      break;
    *(_QWORD *)(a1 + 8 * a2) = *(_QWORD *)result;
    if ( v8 >= v5 )
      goto LABEL_16;
LABEL_14:
    a2 = v8;
  }
  --v8;
  result = (unsigned int *)(a1 + 8 * v8);
  *(_QWORD *)(a1 + 8 * a2) = *(_QWORD *)result;
  if ( v8 < v5 )
    goto LABEL_14;
LABEL_16:
  if ( !v33 )
  {
LABEL_33:
    if ( (a3 - 2) / 2 == v8 )
    {
      v28 = 2 * v8 + 2;
      v29 = *(_QWORD *)(a1 + 8 * v28 - 8);
      v8 = v28 - 1;
      *(_QWORD *)result = v29;
      result = (unsigned int *)(a1 + 8 * v8);
    }
  }
  v19 = (v8 - 1) / 2;
  if ( v8 > v35 )
  {
    v20 = v8;
    while ( 1 )
    {
      v21 = *(_QWORD *)(a5 + 208);
      result = (unsigned int *)(a1 + 8 * v19);
      v22 = 0;
      v23 = *result;
      v24 = result[1];
      v25 = *(_BYTE **)(v21 + 8LL * (unsigned int)(v36 - 1));
      if ( *v25 )
      {
        v22 = 1;
        if ( (unsigned __int8)(*v25 - 4) <= 0x1Eu )
          v22 = (v25[1] != 1) + 2;
      }
      v26 = 0;
      v27 = *(_BYTE **)(v21 + 8LL * (v24 - 1));
      if ( *v27 )
      {
        v26 = 1;
        if ( (unsigned __int8)(*v27 - 4) <= 0x1Eu )
          v26 = (v27[1] != 1) + 2;
      }
      if ( v23 >= v32 && (v23 != v32 || v22 <= v26 && (v22 != v26 || v24 >= v31)) )
        break;
      *(_QWORD *)(a1 + 8 * v20) = *(_QWORD *)result;
      v20 = v19;
      if ( v35 >= v19 )
        goto LABEL_36;
      v19 = (v19 - 1) / 2;
    }
    result = (unsigned int *)(a1 + 8 * v20);
  }
LABEL_36:
  *result = v34;
  result[1] = v36;
  return result;
}

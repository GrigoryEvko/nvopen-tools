// Function: sub_3072EE0
// Address: 0x3072ee0
//
__int64 __fastcall sub_3072EE0(__int64 a1, int a2, __int64 *a3, __int64 a4, int a5)
{
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // rcx
  unsigned __int16 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // r12
  unsigned __int16 v14; // bx
  __int64 v15; // r15
  unsigned int v16; // r13d
  int v17; // edx
  __int64 result; // rax
  __int64 v19; // rax
  int v20; // eax
  unsigned __int64 v21; // rbx
  signed __int64 v22; // rax
  __int64 v23; // r14
  unsigned __int64 v24; // rax
  bool v25; // of
  __int64 *v26; // [rsp+10h] [rbp-70h]
  signed __int64 v27; // [rsp+18h] [rbp-68h]
  unsigned int v29; // [rsp+2Ch] [rbp-54h]
  _BYTE v30[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31; // [rsp+38h] [rbp-48h]
  __int64 v32; // [rsp+40h] [rbp-40h]

  v7 = *(_QWORD *)(a1 + 24);
  v29 = sub_2FEBEF0(v7, a2);
  if ( a5 )
    return 1;
  if ( v29 == 205 )
  {
    v20 = 206;
    if ( (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 > 1 )
      v20 = 205;
    v29 = v20;
  }
  v8 = *a3;
  v27 = 1;
  v9 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), a3, 0);
  v10 = v9;
  v26 = a3;
  v12 = v11;
  while ( 1 )
  {
    LOWORD(v9) = v10;
    sub_2FE6CC0((__int64)v30, *(_QWORD *)(a1 + 24), v8, v9, v12);
    if ( v30[0] == 10 )
      break;
    if ( !v30[0] )
      goto LABEL_9;
    if ( (v30[0] & 0xFB) == 2 )
    {
      v19 = 2 * v27;
      if ( !is_mul_ok(2u, v27) )
      {
        v19 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v27 <= 0 )
          v19 = 0x8000000000000000LL;
      }
      v27 = v19;
    }
    if ( (_WORD)v31 == v10 && ((_WORD)v31 || v32 == v12) )
    {
LABEL_9:
      v13 = (__int64)v26;
      v14 = v10;
      v15 = a1;
      v16 = a2;
      v17 = *((unsigned __int8 *)v26 + 8);
      if ( (unsigned int)(v17 - 17) > 1 )
      {
        if ( !v14 )
          goto LABEL_18;
      }
      else if ( (unsigned __int16)(v14 - 17) > 0xD3u )
      {
        goto LABEL_18;
      }
      goto LABEL_11;
    }
    v9 = v31;
    v12 = v32;
    v10 = v31;
  }
  v13 = (__int64)v26;
  v14 = v10;
  v15 = a1;
  v16 = a2;
  v17 = *((unsigned __int8 *)v26 + 8);
  if ( !v14 )
    v14 = 8;
  if ( (unsigned int)(v17 - 17) <= 1 && (unsigned __int16)(v14 - 17) > 0xD3u )
    goto LABEL_18;
  v27 = 0;
LABEL_11:
  if ( *(_QWORD *)(v7 + 8LL * v14 + 112) && (v29 > 0x1F3 || *(_BYTE *)(v29 + 500LL * v14 + v7 + 6414) != 2) )
    return v27;
LABEL_18:
  if ( v17 == 17 )
  {
    v21 = *(unsigned int *)(v13 + 32);
    if ( a4 && (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 <= 1 )
      a4 = **(_QWORD **)(a4 + 16);
    v22 = sub_3072EE0(v15, v16, **(_QWORD **)(v13 + 16), a4, 0);
    v23 = v22 * v21;
    if ( !is_mul_ok(v22, v21) )
    {
      if ( !v21 || (v23 = 0x7FFFFFFFFFFFFFFFLL, v22 <= 0) )
        v23 = 0x8000000000000000LL;
    }
    v24 = sub_30727B0(v15, v13, 1, 0);
    v25 = __OFADD__(v23, v24);
    result = v23 + v24;
    if ( v25 )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v23 <= 0 )
        return 0x8000000000000000LL;
    }
  }
  else
  {
    return v17 != 18;
  }
  return result;
}

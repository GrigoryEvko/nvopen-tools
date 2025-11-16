// Function: sub_C1C4A0
// Address: 0xc1c4a0
//
_QWORD *__fastcall sub_C1C4A0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  unsigned __int64 v4; // r15
  unsigned int v5; // eax
  _QWORD *v6; // rdx
  _QWORD *v7; // rcx
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rcx
  const void *v11; // rsi
  const void *v12; // rdi
  size_t v13; // rdx
  unsigned int v14; // eax
  bool v15; // cf
  unsigned __int64 v16; // r13
  unsigned __int64 v17; // r12
  const void *v18; // rsi
  const void *v19; // rdi
  size_t v20; // rdx
  int v21; // eax
  __int64 v23; // rax
  unsigned __int64 v24; // r15
  unsigned __int64 v25; // r13
  unsigned __int64 v27; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD **)(a1 + 16);
  if ( !v3 )
  {
    v3 = (_QWORD *)(a1 + 8);
    goto LABEL_33;
  }
  v4 = *(_QWORD *)(a2 + 16);
  while ( 1 )
  {
    v8 = v3[6];
    if ( v8 != v4 )
    {
      LOBYTE(v5) = v8 < v4;
LABEL_4:
      v6 = (_QWORD *)v3[2];
      v7 = (_QWORD *)v3[3];
      if ( (_BYTE)v5 )
        goto LABEL_6;
LABEL_5:
      v6 = v7;
      LOBYTE(v5) = 0;
      goto LABEL_6;
    }
    v9 = v3[5];
    v10 = *(_QWORD *)(a2 + 8);
    v11 = (const void *)v3[4];
    v12 = *(const void **)a2;
    if ( v9 < v10 )
    {
      if ( v12 == v11 )
        goto LABEL_16;
      v13 = v3[5];
      goto LABEL_12;
    }
    if ( v12 != v11 )
    {
      v13 = *(_QWORD *)(a2 + 8);
LABEL_12:
      if ( !v12 )
        goto LABEL_42;
      if ( !v11 )
        goto LABEL_17;
      v27 = *(_QWORD *)(a2 + 8);
      v14 = memcmp(v12, v11, v13);
      v10 = v27;
      if ( v14 )
      {
        v5 = v14 >> 31;
        goto LABEL_4;
      }
    }
    if ( v9 == v10 )
      goto LABEL_17;
LABEL_16:
    if ( v9 <= v10 )
    {
LABEL_17:
      v7 = (_QWORD *)v3[3];
      goto LABEL_5;
    }
LABEL_42:
    v6 = (_QWORD *)v3[2];
    LOBYTE(v5) = 1;
LABEL_6:
    if ( !v6 )
      break;
    v3 = v6;
  }
  if ( !(_BYTE)v5 )
  {
    v15 = v4 < v8;
    if ( v4 == v8 )
      goto LABEL_22;
LABEL_35:
    if ( v15 )
      return 0;
    return v3;
  }
LABEL_33:
  if ( *(_QWORD **)(a1 + 24) != v3 )
  {
    v23 = sub_220EF80(v3);
    v24 = *(_QWORD *)(a2 + 16);
    v25 = *(_QWORD *)(v23 + 48);
    v3 = (_QWORD *)v23;
    v15 = v24 < v25;
    if ( v24 != v25 )
      goto LABEL_35;
LABEL_22:
    v16 = *(_QWORD *)(a2 + 8);
    v17 = v3[5];
    v18 = *(const void **)a2;
    v19 = (const void *)v3[4];
    if ( v16 >= v17 )
    {
      if ( v19 != v18 )
      {
        v20 = v3[5];
        goto LABEL_25;
      }
      goto LABEL_28;
    }
    if ( v19 != v18 )
    {
      v20 = *(_QWORD *)(a2 + 8);
LABEL_25:
      if ( !v19 )
        return 0;
      if ( !v18 )
        return v3;
      v21 = memcmp(v19, v18, v20);
      if ( v21 )
      {
        if ( v21 < 0 )
          return 0;
        return v3;
      }
LABEL_28:
      if ( v16 == v17 )
        return v3;
    }
    if ( v16 > v17 )
      return 0;
    return v3;
  }
  return 0;
}

// Function: sub_2425EA0
// Address: 0x2425ea0
//
int __fastcall sub_2425EA0(__int64 a1, __int64 a2, __int64 a3, size_t *a4)
{
  size_t *v4; // r15
  __int64 v5; // r12
  __int64 i; // r15
  __int64 v7; // rbx
  size_t **v8; // r9
  size_t *v9; // r14
  __int64 v10; // rax
  size_t *v11; // rsi
  size_t v12; // r10
  size_t v13; // r12
  size_t v14; // rdx
  int v15; // eax
  size_t **v16; // r13
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // r14
  __int64 v20; // rbx
  size_t *v21; // r8
  size_t **v22; // r15
  size_t v23; // r9
  size_t *v24; // r11
  size_t v25; // r12
  size_t v26; // rdx
  __int64 v27; // rbx
  size_t *v28; // rax
  void *s2; // [rsp+18h] [rbp-58h]
  void *s2a; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+28h] [rbp-48h]
  size_t *v36; // [rsp+28h] [rbp-48h]
  size_t **v37; // [rsp+30h] [rbp-40h]
  size_t v38; // [rsp+30h] [rbp-40h]
  size_t v39; // [rsp+38h] [rbp-38h]
  size_t *v40; // [rsp+38h] [rbp-38h]

  v4 = a4;
  v5 = a1;
  v35 = (a3 - 1) / 2;
  s2 = (void *)(a3 & 1);
  if ( a2 >= v35 )
  {
    LODWORD(v17) = a2;
    v16 = (size_t **)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_24;
    v7 = a2;
LABEL_27:
    if ( (a3 - 2) / 2 == v7 )
    {
      v27 = 2 * v7 + 2;
      v28 = *(size_t **)(v5 + 8 * v27 - 8);
      v7 = v27 - 1;
      *v16 = v28;
      v16 = (size_t **)(v5 + 8 * v7);
    }
    goto LABEL_15;
  }
  for ( i = a2; ; i = v7 )
  {
    v7 = 2 * (i + 1);
    v10 = 16 * (i + 1);
    v8 = (size_t **)(a1 + v10);
    v11 = *(size_t **)(a1 + v10 - 8);
    v9 = *(size_t **)(a1 + v10);
    v12 = *v11;
    v13 = *v9;
    v14 = *v9;
    if ( *v11 <= *v9 )
      v14 = *v11;
    if ( v14 )
    {
      v37 = (size_t **)(a1 + v10);
      v39 = *v11;
      v15 = memcmp(v9 + 24, v11 + 24, v14);
      v12 = v39;
      v8 = v37;
      if ( v15 )
        break;
    }
    if ( v12 != v13 && v12 > v13 )
      goto LABEL_5;
LABEL_6:
    *(_QWORD *)(a1 + 8 * i) = v9;
    if ( v7 >= v35 )
      goto LABEL_14;
LABEL_7:
    ;
  }
  if ( v15 < 0 )
  {
LABEL_5:
    --v7;
    v8 = (size_t **)(a1 + 8 * v7);
    v9 = *v8;
    goto LABEL_6;
  }
  *(_QWORD *)(a1 + 8 * i) = v9;
  if ( v7 < v35 )
    goto LABEL_7;
LABEL_14:
  v5 = a1;
  v4 = a4;
  v16 = v8;
  if ( !s2 )
    goto LABEL_27;
LABEL_15:
  LODWORD(v17) = v7 - 1;
  if ( v7 > a2 )
  {
    LODWORD(v17) = (_DWORD)v4 + 192;
    v18 = v7;
    v19 = v5;
    v20 = (v7 - 1) / 2;
    s2a = v4 + 24;
    v21 = v4;
    while ( 1 )
    {
      v22 = (size_t **)(v19 + 8 * v20);
      v23 = *v21;
      v24 = *v22;
      v25 = **v22;
      v26 = v25;
      if ( *v21 <= v25 )
        v26 = *v21;
      if ( v26
        && (v36 = v21,
            v38 = *v21,
            v40 = *v22,
            LODWORD(v17) = memcmp(v24 + 24, s2a, v26),
            v24 = v40,
            v23 = v38,
            v21 = v36,
            (_DWORD)v17) )
      {
        if ( (int)v17 >= 0 )
          goto LABEL_23;
      }
      else if ( v23 == v25 || v23 <= v25 )
      {
LABEL_23:
        v4 = v21;
        v16 = (size_t **)(v19 + 8 * v18);
        goto LABEL_24;
      }
      *(_QWORD *)(v19 + 8 * v18) = v24;
      v18 = v20;
      v17 = (v20 - 1) / 2;
      if ( a2 >= v20 )
        break;
      v20 = (v20 - 1) / 2;
    }
    v16 = (size_t **)(v19 + 8 * v20);
    v4 = v21;
  }
LABEL_24:
  *v16 = v4;
  return v17;
}

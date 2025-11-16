// Function: sub_17B7740
// Address: 0x17b7740
//
int __fastcall sub_17B7740(__int64 a1, __int64 a2, __int64 a3, size_t *a4)
{
  size_t *v4; // r15
  __int64 i; // r15
  int v7; // eax
  size_t v8; // r10
  size_t v9; // r9
  __int64 v10; // r14
  size_t **v11; // rbx
  size_t *v12; // r13
  size_t *v13; // rsi
  const void *v14; // rsi
  const void *v15; // rdi
  size_t **v16; // r13
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // r12
  size_t *v21; // r8
  __int64 v22; // r14
  size_t *v23; // r9
  size_t v24; // r11
  size_t **v25; // r15
  size_t v26; // rbx
  const void *v27; // rdi
  __int64 v28; // rbx
  size_t *v29; // rax
  void *s2; // [rsp+18h] [rbp-58h]
  void *s2a; // [rsp+18h] [rbp-58h]
  size_t v35; // [rsp+20h] [rbp-50h]
  size_t v36; // [rsp+20h] [rbp-50h]
  size_t *v37; // [rsp+20h] [rbp-50h]
  size_t *v38; // [rsp+20h] [rbp-50h]
  size_t v39; // [rsp+28h] [rbp-48h]
  size_t v40; // [rsp+28h] [rbp-48h]
  size_t v41; // [rsp+28h] [rbp-48h]
  size_t *v42; // [rsp+28h] [rbp-48h]
  __int64 v43; // [rsp+30h] [rbp-40h]
  size_t *v44; // [rsp+30h] [rbp-40h]
  size_t v45; // [rsp+30h] [rbp-40h]

  v4 = a4;
  v43 = (a3 - 1) / 2;
  s2 = (void *)(a3 & 1);
  if ( a2 < v43 )
  {
    for ( i = a2; ; i = v10 )
    {
      v10 = 2 * (i + 1);
      v11 = (size_t **)(a1 + 16 * (i + 1));
      v13 = *(v11 - 1);
      v12 = *v11;
      v9 = *v13;
      v14 = v13 + 22;
      v8 = **v11;
      v15 = *v11 + 22;
      if ( v9 >= v8 )
      {
        if ( !v8 || (v35 = v9, v39 = **v11, v7 = memcmp(v15, v14, *v12), v8 = v39, v9 = v35, !v7) )
        {
          if ( v9 != v8 )
          {
LABEL_6:
            if ( v9 > v8 )
            {
LABEL_7:
              --v10;
              v11 = (size_t **)(a1 + 8 * v10);
              v12 = *v11;
            }
          }
LABEL_8:
          *(_QWORD *)(a1 + 8 * i) = v12;
          if ( v10 >= v43 )
            goto LABEL_15;
          continue;
        }
      }
      else
      {
        if ( !v9 )
          goto LABEL_8;
        v36 = *v12;
        v40 = v9;
        v7 = memcmp(v15, v14, v9);
        v9 = v40;
        v8 = v36;
        if ( !v7 )
          goto LABEL_6;
      }
      if ( v7 < 0 )
        goto LABEL_7;
      *(_QWORD *)(a1 + 8 * i) = v12;
      if ( v10 >= v43 )
      {
LABEL_15:
        v16 = v11;
        v4 = a4;
        v17 = v10;
        if ( s2 )
          goto LABEL_16;
        goto LABEL_32;
      }
    }
  }
  LODWORD(v18) = a2;
  v16 = (size_t **)(a1 + 8 * a2);
  if ( (a3 & 1) == 0 )
  {
    v17 = a2;
LABEL_32:
    if ( (a3 - 2) / 2 == v17 )
    {
      v28 = 2 * v17 + 2;
      v29 = *(size_t **)(a1 + 8 * v28 - 8);
      v17 = v28 - 1;
      *v16 = v29;
      v16 = (size_t **)(a1 + 8 * v17);
    }
LABEL_16:
    LODWORD(v18) = v17 - 1;
    if ( v17 > a2 )
    {
      v19 = v17;
      s2a = v4 + 22;
      v18 = a1;
      v20 = (v17 - 1) / 2;
      v21 = v4;
      v22 = v18;
      while ( 1 )
      {
        v25 = (size_t **)(v22 + 8 * v20);
        v24 = *v21;
        v23 = *v25;
        v26 = **v25;
        v27 = *v25 + 22;
        if ( *v21 >= v26 )
        {
          if ( !v26
            || (v37 = v21,
                v41 = *v21,
                v44 = *v25,
                LODWORD(v18) = memcmp(v27, s2a, **v25),
                v23 = v44,
                v24 = v41,
                v21 = v37,
                !(_DWORD)v18) )
          {
            if ( v24 == v26 )
              goto LABEL_28;
LABEL_21:
            if ( v24 <= v26 )
              goto LABEL_28;
            goto LABEL_22;
          }
        }
        else
        {
          if ( !v24 )
            goto LABEL_28;
          v38 = v21;
          v42 = *v25;
          v45 = *v21;
          LODWORD(v18) = memcmp(v27, s2a, *v21);
          v24 = v45;
          v23 = v42;
          v21 = v38;
          if ( !(_DWORD)v18 )
            goto LABEL_21;
        }
        if ( (int)v18 >= 0 )
        {
LABEL_28:
          v4 = v21;
          v16 = (size_t **)(v22 + 8 * v19);
          break;
        }
LABEL_22:
        *(_QWORD *)(v22 + 8 * v19) = v23;
        v19 = v20;
        v18 = (v20 - 1) / 2;
        if ( a2 >= v20 )
        {
          v16 = (size_t **)(v22 + 8 * v20);
          v4 = v21;
          break;
        }
        v20 = (v20 - 1) / 2;
      }
    }
  }
  *v16 = v4;
  return v18;
}

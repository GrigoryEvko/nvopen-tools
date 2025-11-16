// Function: sub_16F1780
// Address: 0x16f1780
//
__int64 __fastcall sub_16F1780(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r9
  __int64 v6; // r8
  int v7; // eax
  __int64 v8; // r10
  __int64 *v9; // r11
  size_t v10; // rcx
  __int64 v11; // r14
  __int64 *v12; // rbx
  __int64 v13; // r12
  __int64 v14; // r13
  size_t v15; // r15
  const void *v16; // rdi
  const void *v17; // rsi
  __int64 v18; // r15
  __int64 v19; // rbx
  __int64 i; // r13
  int v21; // eax
  size_t v22; // r9
  __int64 *v23; // rcx
  __int64 v24; // r12
  const void *v25; // rsi
  size_t v26; // r14
  const void *v27; // rdi
  __int64 v30; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+10h] [rbp-70h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  size_t v35; // [rsp+20h] [rbp-60h]
  __int64 *v36; // [rsp+20h] [rbp-60h]
  __int64 *v37; // [rsp+28h] [rbp-58h]
  size_t v38; // [rsp+30h] [rbp-50h]
  __int64 v39; // [rsp+38h] [rbp-48h]
  size_t v40; // [rsp+38h] [rbp-48h]
  size_t v41; // [rsp+38h] [rbp-48h]

  v4 = a1;
  v30 = a3 & 1;
  v39 = (a3 - 1) / 2;
  if ( a2 < v39 )
  {
    v5 = a2;
    v6 = a1;
    while ( 1 )
    {
      v13 = 2 * (v5 + 1);
      v8 = v13 - 1;
      v12 = (__int64 *)(v6 + 16 * (v5 + 1));
      v9 = (__int64 *)(v6 + 8 * (v13 - 1));
      v11 = *v12;
      v14 = *v9;
      v15 = *(_QWORD *)(*v12 + 16);
      v16 = *(const void **)(*v12 + 8);
      v10 = *(_QWORD *)(*v9 + 16);
      v17 = *(const void **)(*v9 + 8);
      if ( v15 <= v10 )
      {
        if ( !v15 )
          goto LABEL_5;
        v31 = v6;
        v33 = v5;
        v35 = *(_QWORD *)(*v9 + 16);
        v37 = (__int64 *)(v6 + 8 * (v13 - 1));
        v7 = memcmp(v16, v17, *(_QWORD *)(*v12 + 16));
        v8 = v13 - 1;
        v9 = v37;
        v10 = v35;
        v5 = v33;
        v6 = v31;
        if ( !v7 )
        {
LABEL_5:
          if ( v15 != v10 )
          {
LABEL_6:
            if ( v15 < v10 )
            {
              v11 = v14;
              v12 = v9;
              v13 = v8;
            }
          }
LABEL_8:
          *(_QWORD *)(v6 + 8 * v5) = v11;
          if ( v13 >= v39 )
            goto LABEL_16;
          goto LABEL_9;
        }
      }
      else
      {
        if ( !v10 )
          goto LABEL_8;
        v32 = v6;
        v34 = v5;
        v36 = (__int64 *)(v6 + 8 * (v13 - 1));
        v38 = *(_QWORD *)(*v9 + 16);
        v7 = memcmp(v16, v17, v38);
        v10 = v38;
        v8 = v13 - 1;
        v9 = v36;
        v5 = v34;
        v6 = v32;
        if ( !v7 )
          goto LABEL_6;
      }
      if ( v7 < 0 )
      {
        v11 = v14;
        v12 = v9;
        v13 = v8;
      }
      *(_QWORD *)(v6 + 8 * v5) = v11;
      if ( v13 >= v39 )
      {
LABEL_16:
        v4 = v6;
        if ( v30 )
          goto LABEL_17;
        goto LABEL_33;
      }
LABEL_9:
      v5 = v13;
    }
  }
  v12 = (__int64 *)(a1 + 8 * a2);
  if ( (a3 & 1) == 0 )
  {
    v13 = a2;
LABEL_33:
    if ( (a3 - 2) / 2 == v13 )
    {
      v13 = 2 * v13 + 1;
      *v12 = *(_QWORD *)(v4 + 8 * v13);
      v12 = (__int64 *)(v4 + 8 * v13);
    }
LABEL_17:
    if ( v13 > a2 )
    {
      v18 = v4;
      v19 = v13;
      for ( i = (v13 - 1) / 2; ; i = (i - 1) / 2 )
      {
        v23 = (__int64 *)(v18 + 8 * i);
        v24 = *v23;
        v22 = *(_QWORD *)(a4 + 16);
        v25 = *(const void **)(a4 + 8);
        v26 = *(_QWORD *)(*v23 + 16);
        v27 = *(const void **)(*v23 + 8);
        if ( v26 <= v22 )
        {
          if ( !v26
            || (v40 = *(_QWORD *)(a4 + 16),
                v21 = memcmp(v27, v25, *(_QWORD *)(*v23 + 16)),
                v22 = v40,
                v23 = (__int64 *)(v18 + 8 * i),
                !v21) )
          {
            if ( v26 == v22 )
              goto LABEL_29;
LABEL_22:
            if ( v26 >= v22 )
              goto LABEL_29;
            goto LABEL_23;
          }
        }
        else
        {
          if ( !v22 )
            goto LABEL_29;
          v41 = *(_QWORD *)(a4 + 16);
          v21 = memcmp(v27, v25, v41);
          v22 = v41;
          v23 = (__int64 *)(v18 + 8 * i);
          if ( !v21 )
            goto LABEL_22;
        }
        if ( v21 >= 0 )
        {
LABEL_29:
          v12 = (__int64 *)(v18 + 8 * v19);
          break;
        }
LABEL_23:
        *(_QWORD *)(v18 + 8 * v19) = v24;
        v19 = i;
        if ( a2 >= i )
        {
          v12 = v23;
          break;
        }
      }
    }
  }
  *v12 = a4;
  return a4;
}

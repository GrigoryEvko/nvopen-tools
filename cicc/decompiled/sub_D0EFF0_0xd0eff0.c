// Function: sub_D0EFF0
// Address: 0xd0eff0
//
__int64 __fastcall sub_D0EFF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 i; // r15
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // r13
  const char *v9; // rax
  void *v10; // rdx
  void *v11; // r11
  bool v12; // cc
  size_t v13; // rdx
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 *v17; // r12
  __int64 v18; // rdi
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // r15
  size_t v23; // rdx
  const char *v24; // rdi
  size_t v25; // rdx
  size_t v26; // r15
  size_t v27; // rdx
  int v28; // eax
  __int64 v30; // rax
  __int64 v32; // [rsp+10h] [rbp-60h]
  const char *s2; // [rsp+18h] [rbp-58h]
  void *v34; // [rsp+20h] [rbp-50h]
  void *v35; // [rsp+20h] [rbp-50h]
  const char *v36; // [rsp+20h] [rbp-50h]
  __int64 v37; // [rsp+28h] [rbp-48h]
  size_t v38; // [rsp+28h] [rbp-48h]

  v37 = (a3 - 1) / 2;
  v32 = a3 & 1;
  if ( a2 >= v37 )
  {
    v17 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_26;
    v16 = a2;
LABEL_29:
    if ( (a3 - 2) / 2 == v16 )
    {
      v30 = *(_QWORD *)(a1 + 8 * (2 * v16 + 2) - 8);
      v16 = 2 * v16 + 1;
      *v17 = v30;
      v17 = (__int64 *)(a1 + 8 * v16);
    }
    goto LABEL_16;
  }
  for ( i = a2; ; i = v16 )
  {
    v16 = 2 * (i + 1);
    v17 = (__int64 *)(a1 + 16 * (i + 1));
    v15 = *v17;
    v18 = *(_QWORD *)(*(v17 - 1) + 8);
    if ( v18 )
      break;
LABEL_10:
    *(_QWORD *)(a1 + 8 * i) = v15;
    if ( v16 >= v37 )
      goto LABEL_15;
LABEL_11:
    ;
  }
  v35 = *(void **)(*v17 + 8);
  if ( !v35 )
    goto LABEL_14;
  v6 = sub_BD5D20(v18);
  v8 = v7;
  s2 = v6;
  v9 = sub_BD5D20((__int64)v35);
  v11 = v10;
  v12 = (unsigned __int64)v10 <= v8;
  v13 = v8;
  if ( v12 )
    v13 = (size_t)v11;
  if ( v13 )
  {
    v34 = v11;
    v14 = memcmp(v9, s2, v13);
    v11 = v34;
    if ( v14 )
    {
      if ( v14 < 0 )
        goto LABEL_14;
      goto LABEL_9;
    }
  }
  if ( v11 == (void *)v8 || (unsigned __int64)v11 >= v8 )
  {
LABEL_9:
    v15 = *v17;
    goto LABEL_10;
  }
LABEL_14:
  --v16;
  v17 = (__int64 *)(a1 + 8 * v16);
  *(_QWORD *)(a1 + 8 * i) = *v17;
  if ( v16 < v37 )
    goto LABEL_11;
LABEL_15:
  if ( !v32 )
    goto LABEL_29;
LABEL_16:
  v19 = (v16 - 1) / 2;
  if ( v16 > a2 )
  {
    while ( 1 )
    {
      v17 = (__int64 *)(a1 + 8 * v19);
      v20 = *v17;
      v21 = *(_QWORD *)(a4 + 8);
      v22 = *(_QWORD *)(*v17 + 8);
      if ( !v21 )
        break;
      if ( v22 )
      {
        v36 = sub_BD5D20(v21);
        v38 = v23;
        v24 = sub_BD5D20(v22);
        v26 = v25;
        v12 = v38 <= v25;
        v27 = v38;
        if ( !v12 )
          v27 = v26;
        if ( v27 && (v28 = memcmp(v24, v36, v27)) != 0 )
        {
          if ( v28 >= 0 )
            break;
        }
        else if ( v38 == v26 || v38 <= v26 )
        {
          break;
        }
        v20 = *v17;
      }
      *(_QWORD *)(a1 + 8 * v16) = v20;
      v16 = v19;
      if ( a2 >= v19 )
        goto LABEL_26;
      v19 = (v19 - 1) / 2;
    }
    v17 = (__int64 *)(a1 + 8 * v16);
  }
LABEL_26:
  *v17 = a4;
  return a4;
}

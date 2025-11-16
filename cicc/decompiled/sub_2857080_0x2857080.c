// Function: sub_2857080
// Address: 0x2857080
//
void __fastcall sub_2857080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  unsigned int v8; // eax
  __int64 *v9; // r12
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 *v13; // r14
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // r12
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 *v23; // r14
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 *v34; // [rsp+8h] [rbp-48h]
  __int64 *v35; // [rsp+8h] [rbp-48h]
  char v36[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v37; // [rsp+18h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 88);
  if ( !v7 )
  {
    v8 = *(_DWORD *)(a1 + 48);
    if ( v8 <= 1 )
      return;
LABEL_9:
    v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL * v8 - 8);
    *(_DWORD *)(a1 + 48) = v8 - 1;
    *(_QWORD *)(a1 + 32) = 1;
    *(_QWORD *)(a1 + 88) = v7;
    goto LABEL_10;
  }
  if ( *(_QWORD *)(a1 + 32) != 1 )
    return;
  if ( !*(_DWORD *)(a1 + 48) )
  {
LABEL_5:
    if ( !*(_DWORD *)(a1 + 52) )
      sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), 1u, 8u, a5, a6);
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL * *(unsigned int *)(a1 + 48)) = v7;
    *(_QWORD *)(a1 + 32) = 0;
    ++*(_DWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 88) = 0;
    return;
  }
  v37 = a2;
  v36[0] = 0;
  sub_2856E20(v7, (__int64)v36);
  if ( v36[0] )
    return;
  v19 = *(__int64 **)(a1 + 40);
  v20 = 8LL * *(unsigned int *)(a1 + 48);
  v35 = &v19[(unsigned __int64)v20 / 8];
  v21 = v20 >> 3;
  v22 = v20 >> 5;
  if ( v22 )
  {
    v23 = &v19[4 * v22];
    while ( 1 )
    {
      v27 = *v19;
      v36[0] = 0;
      v37 = a2;
      sub_2856E20(v27, (__int64)v36);
      if ( v36[0] )
        goto LABEL_29;
      v24 = v19[1];
      v37 = a2;
      sub_2856E20(v24, (__int64)v36);
      if ( v36[0] )
      {
        ++v19;
        goto LABEL_29;
      }
      v25 = v19[2];
      v37 = a2;
      sub_2856E20(v25, (__int64)v36);
      if ( v36[0] )
      {
        v19 += 2;
        goto LABEL_29;
      }
      v26 = v19[3];
      v37 = a2;
      sub_2856E20(v26, (__int64)v36);
      if ( v36[0] )
      {
        v19 += 3;
        goto LABEL_29;
      }
      v19 += 4;
      if ( v23 == v19 )
      {
        v21 = v35 - v19;
        break;
      }
    }
  }
  if ( v21 == 2 )
    goto LABEL_48;
  if ( v21 == 3 )
  {
    v29 = *v19;
    v36[0] = 0;
    v37 = a2;
    sub_2856E20(v29, (__int64)v36);
    if ( v36[0] )
      goto LABEL_29;
    ++v19;
LABEL_48:
    v30 = *v19;
    v36[0] = 0;
    v37 = a2;
    sub_2856E20(v30, (__int64)v36);
    if ( v36[0] )
      goto LABEL_29;
    ++v19;
    goto LABEL_50;
  }
  if ( v21 != 1 )
    return;
LABEL_50:
  v31 = *v19;
  v36[0] = 0;
  v37 = a2;
  sub_2856E20(v31, (__int64)v36);
  if ( !v36[0] )
    return;
LABEL_29:
  if ( v35 == v19 )
    return;
  v8 = *(_DWORD *)(a1 + 48);
  v7 = *(_QWORD *)(a1 + 88);
  if ( !v8 )
    goto LABEL_5;
  if ( !v7 )
    goto LABEL_9;
LABEL_10:
  v36[0] = 0;
  v37 = a2;
  sub_2856E20(v7, (__int64)v36);
  if ( v36[0] )
    return;
  v9 = *(__int64 **)(a1 + 40);
  v10 = 8LL * *(unsigned int *)(a1 + 48);
  v34 = &v9[(unsigned __int64)v10 / 8];
  v11 = v10 >> 3;
  v12 = v10 >> 5;
  if ( v12 )
  {
    v13 = &v9[4 * v12];
    while ( 1 )
    {
      v17 = *v9;
      v36[0] = 0;
      v37 = a2;
      sub_2856E20(v17, (__int64)v36);
      if ( v36[0] )
        goto LABEL_18;
      v14 = v9[1];
      v37 = a2;
      sub_2856E20(v14, (__int64)v36);
      if ( v36[0] )
      {
        v34 = v9 + 1;
        goto LABEL_19;
      }
      v15 = v9[2];
      v37 = a2;
      sub_2856E20(v15, (__int64)v36);
      if ( v36[0] )
      {
        v34 = v9 + 2;
        goto LABEL_19;
      }
      v16 = v9[3];
      v37 = a2;
      sub_2856E20(v16, (__int64)v36);
      if ( v36[0] )
      {
        v34 = v9 + 3;
        goto LABEL_19;
      }
      v9 += 4;
      if ( v13 == v9 )
      {
        v11 = v34 - v9;
        break;
      }
    }
  }
  switch ( v11 )
  {
    case 2LL:
      goto LABEL_59;
    case 3LL:
      v32 = *v9;
      v36[0] = 0;
      v37 = a2;
      sub_2856E20(v32, (__int64)v36);
      if ( v36[0] )
        goto LABEL_18;
      ++v9;
LABEL_59:
      v33 = *v9;
      v36[0] = 0;
      v37 = a2;
      sub_2856E20(v33, (__int64)v36);
      if ( !v36[0] )
      {
        ++v9;
        goto LABEL_37;
      }
LABEL_18:
      v34 = v9;
      break;
    case 1LL:
LABEL_37:
      v28 = *v9;
      v36[0] = 0;
      v37 = a2;
      sub_2856E20(v28, (__int64)v36);
      if ( !v36[0] )
        v9 = v34;
      v34 = v9;
      break;
  }
LABEL_19:
  if ( v34 != (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * *(unsigned int *)(a1 + 48)) )
  {
    v18 = *(_QWORD *)(a1 + 88);
    *(_QWORD *)(a1 + 88) = *v34;
    *v34 = v18;
  }
}

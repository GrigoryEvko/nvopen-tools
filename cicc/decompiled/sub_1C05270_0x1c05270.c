// Function: sub_1C05270
// Address: 0x1c05270
//
__int64 __fastcall sub_1C05270(__int64 a1, __int64 a2, __int64 a3)
{
  char *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 v10; // r8
  __int64 v11; // rdx
  int v12; // r11d
  unsigned int v13; // r14d
  unsigned int v14; // edi
  unsigned int v15; // r10d
  __int64 *v16; // rax
  __int64 *v17; // rcx
  __int64 *v18; // r9
  __int64 *v19; // rax
  int v21; // r15d
  __int64 **v22; // r11
  int v23; // r15d
  __int64 v24; // rdi
  int v25; // eax
  int v26; // edx
  int v27; // eax
  int v28; // eax
  __int64 v29; // rsi
  unsigned int v30; // r14d
  __int64 *v31; // rdi
  int v32; // eax
  int v33; // eax
  __int64 v34; // rsi
  unsigned int v35; // r14d
  __int64 **v36; // r10
  const char *v37; // [rsp+0h] [rbp-50h] BYREF
  __int64 v38; // [rsp+8h] [rbp-48h]
  char v39[64]; // [rsp+10h] [rbp-40h] BYREF

  sub_223E0D0(a2, "\"", 1);
  v5 = (char *)sub_1649960(a3);
  if ( v5 )
  {
    v37 = v39;
    sub_1C04B10((__int64 *)&v37, v5, (__int64)&v5[v6]);
    v7 = sub_223E0D0(a2, v37, v38);
  }
  else
  {
    v38 = 0;
    v37 = v39;
    v39[0] = 0;
    v7 = sub_223E0D0(a2, v39, 0);
  }
  sub_223E0D0(v7, "\"", 1);
  if ( v37 != v39 )
    j_j___libc_free_0(v37, *(_QWORD *)v39 + 1LL);
  v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 160) + 8LL) + 104LL);
  v9 = *(unsigned int *)(v8 + 64);
  if ( !(_DWORD)v9 )
  {
LABEL_13:
    sub_223E0D0(a2, " [style=dotted]", 15);
    return sub_223E0D0(a2, ";\n", 2);
  }
  v10 = (unsigned int)(v9 - 1);
  v11 = *(_QWORD *)(v8 + 48);
  v12 = 1;
  v13 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v14 = v10 & v13;
  v15 = v10 & v13;
  v16 = (__int64 *)(v11 + 16LL * ((unsigned int)v10 & v13));
  v17 = (__int64 *)*v16;
  v18 = (__int64 *)*v16;
  if ( a3 == *v16 )
  {
    if ( v16 != (__int64 *)(v11 + 16 * v9) )
    {
      v19 = (__int64 *)v16[1];
      goto LABEL_9;
    }
    goto LABEL_13;
  }
  while ( 1 )
  {
    if ( v18 == (__int64 *)-8LL )
      goto LABEL_13;
    v21 = v12 + 1;
    v15 = v10 & (v12 + v15);
    v22 = (__int64 **)(v11 + 16LL * v15);
    v18 = *v22;
    if ( (__int64 *)a3 == *v22 )
      break;
    v12 = v21;
  }
  v23 = 1;
  v18 = 0;
  if ( v22 == (__int64 **)(v11 + 16LL * (unsigned int)v9) )
    goto LABEL_13;
  while ( v17 != (__int64 *)-8LL )
  {
    if ( v17 != (__int64 *)-16LL || v18 )
      v16 = v18;
    v18 = (__int64 *)(unsigned int)(v23 + 1);
    v14 = v10 & (v23 + v14);
    v36 = (__int64 **)(v11 + 16LL * v14);
    v17 = *v36;
    if ( (__int64 *)a3 == *v36 )
    {
      v19 = v36[1];
      goto LABEL_9;
    }
    ++v23;
    v18 = v16;
    v16 = (__int64 *)(v11 + 16LL * v14);
  }
  v10 = (unsigned int)(2 * v9);
  v24 = v8 + 40;
  if ( !v18 )
    v18 = v16;
  v25 = *(_DWORD *)(v8 + 56);
  ++*(_QWORD *)(v8 + 40);
  v26 = v25 + 1;
  if ( 4 * (v25 + 1) >= (unsigned int)(3 * v9) )
  {
    sub_1C04E30(v24, v10);
    v27 = *(_DWORD *)(v8 + 64);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(v8 + 48);
      v30 = v28 & v13;
      v26 = *(_DWORD *)(v8 + 56) + 1;
      v18 = (__int64 *)(v29 + 16LL * v30);
      v17 = (__int64 *)*v18;
      if ( a3 == *v18 )
        goto LABEL_23;
      v10 = 1;
      v31 = 0;
      while ( v17 != (__int64 *)-8LL )
      {
        if ( !v31 && v17 == (__int64 *)-16LL )
          v31 = v18;
        v30 = v28 & (v10 + v30);
        v18 = (__int64 *)(v29 + 16LL * v30);
        v17 = (__int64 *)*v18;
        if ( a3 == *v18 )
          goto LABEL_23;
        v10 = (unsigned int)(v10 + 1);
      }
LABEL_30:
      if ( v31 )
        v18 = v31;
      goto LABEL_23;
    }
LABEL_52:
    ++*(_DWORD *)(v8 + 56);
    BUG();
  }
  v17 = (__int64 *)((unsigned int)v9 >> 3);
  if ( (int)v9 - *(_DWORD *)(v8 + 60) - v26 <= (unsigned int)v17 )
  {
    sub_1C04E30(v24, v9);
    v32 = *(_DWORD *)(v8 + 64);
    if ( v32 )
    {
      v33 = v32 - 1;
      v34 = *(_QWORD *)(v8 + 48);
      v31 = 0;
      v35 = v33 & v13;
      v10 = 1;
      v26 = *(_DWORD *)(v8 + 56) + 1;
      v18 = (__int64 *)(v34 + 16LL * v35);
      v17 = (__int64 *)*v18;
      if ( a3 == *v18 )
        goto LABEL_23;
      while ( v17 != (__int64 *)-8LL )
      {
        if ( !v31 && v17 == (__int64 *)-16LL )
          v31 = v18;
        v35 = v33 & (v10 + v35);
        v18 = (__int64 *)(v34 + 16LL * v35);
        v17 = (__int64 *)*v18;
        if ( a3 == *v18 )
          goto LABEL_23;
        v10 = (unsigned int)(v10 + 1);
      }
      goto LABEL_30;
    }
    goto LABEL_52;
  }
LABEL_23:
  *(_DWORD *)(v8 + 56) = v26;
  if ( *v18 != -8 )
    --*(_DWORD *)(v8 + 60);
  *v18 = a3;
  v19 = 0;
  v18[1] = 0;
LABEL_9:
  if ( *((_DWORD *)v19 + 3) )
    goto LABEL_13;
  sub_223E0D0(a2, " [style=filled, fillcolor=red, fontcolor=white]", 47, v17, v10, v18);
  return sub_223E0D0(a2, ";\n", 2);
}

// Function: sub_F83200
// Address: 0xf83200
//
__int64 __fastcall sub_F83200(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r11d
  __int64 *v8; // r9
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r10
  __int64 v12; // r12
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // rdi
  unsigned __int16 v17; // ax
  __int64 *v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // r15
  __int64 v22; // rax
  unsigned int v23; // esi
  __int64 v24; // rdi
  __int64 v25; // r8
  __int64 *v26; // r10
  int v27; // r14d
  unsigned int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rdx
  int v35; // eax
  __int64 v36; // rsi
  int v37; // edi
  unsigned int v38; // edx
  __int64 *v39; // rax
  __int64 v40; // r8
  int v41; // eax
  int v42; // edx
  int v43; // eax
  int v44; // r10d
  __int64 *i; // [rsp+0h] [rbp-60h]
  __int64 v46; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v47; // [rsp+18h] [rbp-48h] BYREF
  __int64 *v48; // [rsp+20h] [rbp-40h] BYREF
  __int64 v49; // [rsp+28h] [rbp-38h]

  v2 = a2;
  v3 = a1 + 384;
  v46 = a2;
  v48 = (__int64 *)a2;
  v5 = *(_DWORD *)(a1 + 408);
  v49 = 0;
  if ( v5 )
  {
    v6 = *(_QWORD *)(a1 + 392);
    v7 = 1;
    v8 = 0;
    v9 = (v5 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v2 == *v10 )
      return v10[1];
    while ( v11 != -4096 )
    {
      if ( v11 == -8192 && !v8 )
        v8 = v10;
      v9 = (v5 - 1) & (v7 + v9);
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( v2 == *v10 )
        return v10[1];
      ++v7;
    }
    v14 = *(_DWORD *)(a1 + 400);
    if ( !v8 )
      v8 = v10;
    ++*(_QWORD *)(a1 + 384);
    v15 = v14 + 1;
    v47 = v8;
    if ( 4 * v15 < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 404) - v15 > v5 >> 3 )
        goto LABEL_15;
      goto LABEL_35;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 384);
    v47 = 0;
  }
  v5 *= 2;
LABEL_35:
  sub_F83020(v3, v5);
  sub_F81B60(v3, (__int64 *)&v48, &v47);
  v2 = (__int64)v48;
  v8 = v47;
  v15 = *(_DWORD *)(a1 + 400) + 1;
LABEL_15:
  *(_DWORD *)(a1 + 400) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 404);
  *v8 = v2;
  v16 = v46;
  v8[1] = v49;
  v17 = *(_WORD *)(v16 + 24);
  if ( v17 <= 0xEu )
  {
    v12 = 0;
    if ( v17 <= 1u )
      return v12;
    if ( v17 == 8 )
      v12 = *(_QWORD *)(v16 + 48);
    v18 = (__int64 *)sub_D960E0(v16);
    for ( i = &v18[v19]; i != v18; v12 = sub_F79730(v12, v22, v21) )
    {
      v20 = *v18++;
      v21 = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
      v22 = sub_F83200(a1, v20);
    }
    v23 = *(_DWORD *)(a1 + 408);
    if ( v23 )
    {
      v24 = v46;
      v25 = *(_QWORD *)(a1 + 392);
      v26 = 0;
      v27 = 1;
      v28 = (v23 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
      v29 = (__int64 *)(v25 + 16LL * v28);
      v30 = *v29;
      if ( v46 == *v29 )
      {
LABEL_25:
        v31 = v29 + 1;
LABEL_26:
        *v31 = v12;
        return v12;
      }
      while ( v30 != -4096 )
      {
        if ( !v26 && v30 == -8192 )
          v26 = v29;
        v28 = (v23 - 1) & (v27 + v28);
        v29 = (__int64 *)(v25 + 16LL * v28);
        v30 = *v29;
        if ( v46 == *v29 )
          goto LABEL_25;
        ++v27;
      }
      if ( !v26 )
        v26 = v29;
      v41 = *(_DWORD *)(a1 + 400);
      ++*(_QWORD *)(a1 + 384);
      v42 = v41 + 1;
      v48 = v26;
      if ( 4 * (v41 + 1) < 3 * v23 )
      {
        if ( v23 - *(_DWORD *)(a1 + 404) - v42 > v23 >> 3 )
        {
LABEL_46:
          *(_DWORD *)(a1 + 400) = v42;
          if ( *v26 != -4096 )
            --*(_DWORD *)(a1 + 404);
          *v26 = v24;
          v31 = v26 + 1;
          v26[1] = 0;
          goto LABEL_26;
        }
LABEL_51:
        sub_F83020(v3, v23);
        sub_F81B60(v3, &v46, &v48);
        v24 = v46;
        v26 = v48;
        v42 = *(_DWORD *)(a1 + 400) + 1;
        goto LABEL_46;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 384);
      v48 = 0;
    }
    v23 *= 2;
    goto LABEL_51;
  }
  if ( v17 != 15 )
    BUG();
  v32 = *(_QWORD *)(v16 - 8);
  v12 = 0;
  if ( *(_BYTE *)v32 > 0x1Cu )
  {
    v33 = *(_QWORD *)(v32 + 40);
    v34 = *(_QWORD *)(*(_QWORD *)a1 + 48LL);
    v35 = *(_DWORD *)(v34 + 24);
    v36 = *(_QWORD *)(v34 + 8);
    if ( v35 )
    {
      v37 = v35 - 1;
      v38 = (v35 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v39 = (__int64 *)(v36 + 16LL * v38);
      v40 = *v39;
      if ( v33 == *v39 )
      {
LABEL_31:
        v12 = v39[1];
      }
      else
      {
        v43 = 1;
        while ( v40 != -4096 )
        {
          v44 = v43 + 1;
          v38 = v37 & (v43 + v38);
          v39 = (__int64 *)(v36 + 16LL * v38);
          v40 = *v39;
          if ( v33 == *v39 )
            goto LABEL_31;
          v43 = v44;
        }
        v12 = 0;
      }
    }
    v8[1] = v12;
  }
  return v12;
}

// Function: sub_18DD900
// Address: 0x18dd900
//
__int64 __fastcall sub_18DD900(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  int v6; // eax
  __int64 v7; // rcx
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r12
  int v14; // eax
  __int64 i; // rdi
  int v16; // edi
  __int64 v17; // rax
  unsigned __int8 v18; // al
  int v19; // eax
  unsigned int v20; // esi
  __int64 v21; // rcx
  unsigned int v22; // edx
  __int64 *v23; // r14
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r13
  int v27; // eax
  int v28; // ecx
  __int64 v29; // rdi
  unsigned int v30; // eax
  int v31; // edx
  __int64 v32; // rsi
  int v33; // r8d
  int v34; // r9d
  __int64 *v35; // r8
  int v36; // eax
  int v37; // eax
  int v38; // eax
  __int64 v39; // rsi
  __int64 *v40; // rdi
  unsigned int v41; // r15d
  int v42; // r8d
  __int64 v43; // rcx
  int v44; // r9d
  __int64 *v45; // r8
  unsigned __int64 v46[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v47; // [rsp+10h] [rbp-40h]

  v6 = *(_DWORD *)(a3 + 24);
  if ( v6 )
  {
    v7 = *(_QWORD *)(a3 + 8);
    v8 = v6 - 1;
    v9 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v10 = (__int64 *)(v7 + 32LL * v9);
    v11 = *v10;
    if ( a1 == *v10 )
    {
LABEL_3:
      v12 = v10[3];
      v46[0] = 6;
      v46[1] = 0;
      v47 = v12;
      if ( v12 != 0 && v12 != -8 && v12 != -16 )
      {
        sub_1649AC0(v46, v10[1] & 0xFFFFFFFFFFFFFFF8LL);
        v12 = v47;
      }
      if ( v12 )
      {
        if ( v12 != -16 && v12 != -8 )
          sub_1649B30(v46);
        return v12;
      }
    }
    else
    {
      v14 = 1;
      while ( v11 != -8 )
      {
        v33 = v14 + 1;
        v9 = v8 & (v14 + v9);
        v10 = (__int64 *)(v7 + 32LL * v9);
        v11 = *v10;
        if ( *v10 == a1 )
          goto LABEL_3;
        v14 = v33;
      }
    }
  }
  for ( i = a1; ; i = *(_QWORD *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)) )
  {
    v17 = sub_14AD280(i, a2, 6u);
    v16 = 23;
    v12 = v17;
    v18 = *(_BYTE *)(v17 + 16);
    if ( v18 > 0x17u )
    {
      if ( v18 != 78 )
      {
        v16 = 2 * (v18 != 29) + 21;
        goto LABEL_15;
      }
      v16 = 21;
      if ( !*(_BYTE *)(*(_QWORD *)(v12 - 24) + 16LL) )
        break;
    }
LABEL_15:
    if ( !(unsigned __int8)sub_1439C90(v16) )
      goto LABEL_21;
LABEL_16:
    ;
  }
  v19 = sub_1438F00(*(_QWORD *)(v12 - 24));
  if ( (unsigned __int8)sub_1439C90(v19) )
    goto LABEL_16;
LABEL_21:
  v20 = *(_DWORD *)(a3 + 24);
  if ( v20 )
  {
    v21 = *(_QWORD *)(a3 + 8);
    v22 = (v20 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v23 = (__int64 *)(v21 + 32LL * v22);
    v24 = *v23;
    if ( *v23 == a1 )
    {
LABEL_23:
      v25 = v23[3];
      if ( v25 != v12 )
      {
        v26 = (__int64)(v23 + 1);
        if ( v25 != 0 && v25 != -16 && v25 != -8 )
          sub_1649B30(v23 + 1);
        goto LABEL_27;
      }
      return v12;
    }
    v34 = 1;
    v35 = 0;
    while ( v24 != -8 )
    {
      if ( v24 == -16 && !v35 )
        v35 = v23;
      v22 = (v20 - 1) & (v34 + v22);
      v23 = (__int64 *)(v21 + 32LL * v22);
      v24 = *v23;
      if ( *v23 == a1 )
        goto LABEL_23;
      ++v34;
    }
    v36 = *(_DWORD *)(a3 + 16);
    if ( v35 )
      v23 = v35;
    ++*(_QWORD *)a3;
    v31 = v36 + 1;
    if ( 4 * (v36 + 1) < 3 * v20 )
    {
      if ( v20 - *(_DWORD *)(a3 + 20) - v31 > v20 >> 3 )
        goto LABEL_33;
      sub_18DD6C0(a3, v20);
      v37 = *(_DWORD *)(a3 + 24);
      if ( v37 )
      {
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a3 + 8);
        v40 = 0;
        v41 = v38 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v42 = 1;
        v31 = *(_DWORD *)(a3 + 16) + 1;
        v23 = (__int64 *)(v39 + 32LL * v41);
        v43 = *v23;
        if ( *v23 != a1 )
        {
          while ( v43 != -8 )
          {
            if ( v43 == -16 && !v40 )
              v40 = v23;
            v41 = v38 & (v42 + v41);
            v23 = (__int64 *)(v39 + 32LL * v41);
            v43 = *v23;
            if ( *v23 == a1 )
              goto LABEL_33;
            ++v42;
          }
          if ( v40 )
            v23 = v40;
        }
        goto LABEL_33;
      }
LABEL_70:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)a3;
  }
  sub_18DD6C0(a3, 2 * v20);
  v27 = *(_DWORD *)(a3 + 24);
  if ( !v27 )
    goto LABEL_70;
  v28 = v27 - 1;
  v29 = *(_QWORD *)(a3 + 8);
  v30 = (v27 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v31 = *(_DWORD *)(a3 + 16) + 1;
  v23 = (__int64 *)(v29 + 32LL * v30);
  v32 = *v23;
  if ( *v23 != a1 )
  {
    v44 = 1;
    v45 = 0;
    while ( v32 != -8 )
    {
      if ( v32 == -16 && !v45 )
        v45 = v23;
      v30 = v28 & (v44 + v30);
      v23 = (__int64 *)(v29 + 32LL * v30);
      v32 = *v23;
      if ( *v23 == a1 )
        goto LABEL_33;
      ++v44;
    }
    if ( v45 )
      v23 = v45;
  }
LABEL_33:
  *(_DWORD *)(a3 + 16) = v31;
  if ( *v23 != -8 )
    --*(_DWORD *)(a3 + 20);
  *v23 = a1;
  v26 = (__int64)(v23 + 1);
  v23[1] = 6;
  v23[2] = 0;
  v23[3] = 0;
LABEL_27:
  v23[3] = v12;
  if ( v12 != -8 && v12 != -16 )
    sub_164C220(v26);
  return v12;
}

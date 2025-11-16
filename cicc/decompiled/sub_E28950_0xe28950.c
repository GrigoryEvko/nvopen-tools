// Function: sub_E28950
// Address: 0xe28950
//
unsigned __int64 __fastcall sub_E28950(__int64 a1, size_t *a2)
{
  __int16 v2; // r14
  __int16 v3; // ax
  __int16 v4; // r14
  unsigned __int64 v5; // rbx
  _QWORD *v6; // rdx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r13
  char v9; // al
  _QWORD *v10; // rdx
  unsigned __int64 result; // rax
  _QWORD *v12; // rax
  __int64 *v13; // rax
  __int64 *v14; // r15
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 *v17; // rax
  unsigned __int64 *v18; // rbx
  unsigned __int64 v19; // rdx
  _QWORD *v20; // rdx
  unsigned __int64 v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax

  v2 = ((unsigned __int8)sub_E20730(a2, 4u, "$$J0") != 0) << 7;
  if ( !*a2 )
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  v3 = sub_E22B50(a1, (__int64 *)a2);
  v4 = v3 | v2;
  if ( (v3 & 0x800) == 0 )
  {
    v5 = 0;
    if ( (v4 & 0x200) == 0 )
    {
      if ( (v4 & 0x100) != 0 )
        goto LABEL_5;
LABEL_17:
      v8 = sub_E28570(a1, (__int64 *)a2, (v4 & 0x18) == 0);
      if ( !*(_BYTE *)(a1 + 8) )
        goto LABEL_8;
      return 0;
    }
    v20 = *(_QWORD **)(a1 + 16);
    v21 = (*v20 + v20[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v20[1] = v21 - *v20 + 80;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v26 = (__int64 *)sub_22077B0(32);
      v27 = v26;
      if ( v26 )
      {
        *v26 = 0;
        v26[1] = 0;
        v26[2] = 0;
        v26[3] = 0;
      }
      v28 = sub_2207820(4096);
      v27[2] = 4096;
      *v27 = v28;
      v5 = v28;
      v29 = *(_QWORD *)(a1 + 16);
      v27[1] = 80;
      v27[3] = v29;
      *(_QWORD *)(a1 + 16) = v27;
      if ( !v5 )
        goto LABEL_32;
    }
    else
    {
      if ( !v21 )
      {
LABEL_32:
        if ( (v4 & 0x400) != 0 )
        {
          *(_DWORD *)(v5 + 64) = sub_E21AC0(a1, a2);
          *(_DWORD *)(v5 + 68) = sub_E21AC0(a1, a2);
        }
        *(_DWORD *)(v5 + 72) = sub_E21AC0(a1, a2);
        *(_DWORD *)(v5 + 60) = sub_E21AC0(a1, a2);
        goto LABEL_16;
      }
      v5 = v21;
    }
    *(_BYTE *)(v5 + 12) = 0;
    *(_DWORD *)(v5 + 8) = 13;
    *(_DWORD *)(v5 + 16) = 0;
    *(_BYTE *)(v5 + 20) = 0;
    *(_WORD *)(v5 + 22) = 8;
    *(_DWORD *)(v5 + 24) = 0;
    *(_QWORD *)(v5 + 32) = 0;
    *(_BYTE *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 48) = 0;
    *(_BYTE *)(v5 + 56) = 0;
    *(_QWORD *)v5 = &unk_49E10B0;
    *(_QWORD *)(v5 + 60) = 0;
    *(_QWORD *)(v5 + 68) = 0;
    goto LABEL_32;
  }
  v12 = *(_QWORD **)(a1 + 16);
  v5 = (*v12 + v12[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v12[1] = v5 - *v12 + 80;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v13 = (__int64 *)sub_22077B0(32);
    v14 = v13;
    if ( v13 )
    {
      *v13 = 0;
      v13[1] = 0;
      v13[2] = 0;
      v13[3] = 0;
    }
    v15 = sub_2207820(4096);
    v14[2] = 4096;
    *v14 = v15;
    v5 = v15;
    v16 = *(_QWORD *)(a1 + 16);
    v14[1] = 80;
    v14[3] = v16;
    *(_QWORD *)(a1 + 16) = v14;
  }
  if ( !v5 )
  {
    sub_E21AC0(a1, a2);
    MEMORY[0x3C] = 0;
    BUG();
  }
  *(_BYTE *)(v5 + 12) = 0;
  *(_DWORD *)(v5 + 8) = 13;
  *(_DWORD *)(v5 + 16) = 0;
  *(_BYTE *)(v5 + 20) = 0;
  *(_WORD *)(v5 + 22) = 8;
  *(_DWORD *)(v5 + 24) = 0;
  *(_QWORD *)(v5 + 32) = 0;
  *(_BYTE *)(v5 + 40) = 0;
  *(_QWORD *)(v5 + 48) = 0;
  *(_BYTE *)(v5 + 56) = 0;
  *(_QWORD *)v5 = &unk_49E10B0;
  *(_QWORD *)(v5 + 60) = 0;
  *(_QWORD *)(v5 + 68) = 0;
  *(_DWORD *)(v5 + 60) = sub_E21AC0(a1, a2);
LABEL_16:
  if ( (v4 & 0x100) == 0 )
    goto LABEL_17;
LABEL_5:
  v6 = *(_QWORD **)(a1 + 16);
  v7 = (*v6 + v6[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v6[1] = v7 - *v6 + 64;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v8 = 0;
    if ( !v7 )
      goto LABEL_7;
    v8 = v7;
    goto LABEL_21;
  }
  v22 = (__int64 *)sub_22077B0(32);
  v23 = v22;
  if ( v22 )
  {
    *v22 = 0;
    v22[1] = 0;
    v22[2] = 0;
    v22[3] = 0;
  }
  v24 = sub_2207820(4096);
  v23[2] = 4096;
  *v23 = v24;
  v8 = v24;
  v25 = *(_QWORD *)(a1 + 16);
  v23[1] = 64;
  v23[3] = v25;
  *(_QWORD *)(a1 + 16) = v23;
  if ( v8 )
  {
LABEL_21:
    *(_DWORD *)(v8 + 8) = 3;
    *(_BYTE *)(v8 + 12) = 0;
    *(_DWORD *)(v8 + 16) = 0;
    *(_QWORD *)v8 = &unk_49E1078;
    *(_BYTE *)(v8 + 20) = 0;
    *(_WORD *)(v8 + 22) = 8;
    *(_DWORD *)(v8 + 24) = 0;
    *(_QWORD *)(v8 + 32) = 0;
    *(_BYTE *)(v8 + 40) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_BYTE *)(v8 + 56) = 0;
  }
LABEL_7:
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
LABEL_8:
  if ( v5 )
  {
    *(_DWORD *)(v5 + 8) = *(_DWORD *)(v8 + 8);
    *(_BYTE *)(v5 + 12) = *(_BYTE *)(v8 + 12);
    *(_DWORD *)(v5 + 16) = *(_DWORD *)(v8 + 16);
    *(_BYTE *)(v5 + 20) = *(_BYTE *)(v8 + 20);
    *(_WORD *)(v5 + 22) = *(_WORD *)(v8 + 22);
    *(_DWORD *)(v5 + 24) = *(_DWORD *)(v8 + 24);
    *(_QWORD *)(v5 + 32) = *(_QWORD *)(v8 + 32);
    *(_BYTE *)(v5 + 40) = *(_BYTE *)(v8 + 40);
    *(_QWORD *)(v5 + 48) = *(_QWORD *)(v8 + 48);
    v9 = *(_BYTE *)(v8 + 56);
    v8 = v5;
    *(_BYTE *)(v5 + 56) = v9;
  }
  *(_WORD *)(v8 + 22) = v4;
  v10 = *(_QWORD **)(a1 + 16);
  result = (*v10 + v10[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v10[1] = result - *v10 + 32;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v17 = (unsigned __int64 *)sub_22077B0(32);
    v18 = v17;
    if ( v17 )
    {
      *v17 = 0;
      v17[1] = 0;
      v17[2] = 0;
      v17[3] = 0;
    }
    result = sub_2207820(4096);
    v19 = *(_QWORD *)(a1 + 16);
    v18[2] = 4096;
    *v18 = result;
    v18[3] = v19;
    *(_QWORD *)(a1 + 16) = v18;
    v18[1] = 32;
  }
  if ( !result )
  {
    MEMORY[0x18] = v8;
    BUG();
  }
  *(_QWORD *)(result + 24) = 0;
  *(_DWORD *)(result + 8) = 26;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)result = &unk_49E11B8;
  *(_QWORD *)(result + 24) = v8;
  return result;
}

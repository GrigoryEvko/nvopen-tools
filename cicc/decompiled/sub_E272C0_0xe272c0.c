// Function: sub_E272C0
// Address: 0xe272c0
//
unsigned __int64 __fastcall sub_E272C0(__int64 a1, __int64 *a2)
{
  char *v3; // rdx
  char v4; // al
  __int64 v5; // rsi
  _QWORD *v6; // rdx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r12
  _QWORD *v10; // rdx
  unsigned __int64 v11; // rax
  _QWORD *v12; // rdx
  unsigned __int64 v13; // rax
  __int64 *v14; // rax
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rax
  __int64 *v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rdx
  unsigned __int64 v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rax

  v3 = (char *)a2[1];
  v4 = *v3;
  a2[1] = (__int64)(v3 + 1);
  v5 = *a2;
  *a2 = v5 - 1;
  if ( v4 == 86 )
  {
    v12 = *(_QWORD **)(a1 + 16);
    v13 = (*v12 + v12[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v12[1] = v13 - *v12 + 32;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v14 = (__int64 *)sub_22077B0(32);
      v15 = v14;
      if ( v14 )
      {
        *v14 = 0;
        v14[1] = 0;
        v14[2] = 0;
        v14[3] = 0;
      }
      v16 = sub_2207820(4096);
      v15[2] = 4096;
      *v15 = v16;
      v8 = v16;
      v17 = *(_QWORD *)(a1 + 16);
      v15[1] = 32;
      v15[3] = v17;
      *(_QWORD *)(a1 + 16) = v15;
      if ( !v8 )
        goto LABEL_41;
    }
    else
    {
      if ( !v13 )
        goto LABEL_41;
      v8 = v13;
    }
    *(_BYTE *)(v8 + 12) = 0;
    *(_DWORD *)(v8 + 8) = 15;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)v8 = &unk_49E1120;
    *(_DWORD *)(v8 + 24) = 0;
    goto LABEL_21;
  }
  if ( v4 > 86 )
  {
    if ( v4 != 87 )
      goto LABEL_41;
    if ( v5 == 1 || v3[1] != 52 )
    {
      *(_BYTE *)(a1 + 8) = 1;
      return 0;
    }
    *a2 = v5 - 2;
    a2[1] = (__int64)(v3 + 2);
    v26 = *(_QWORD **)(a1 + 16);
    v27 = (*v26 + v26[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v26[1] = v27 - *v26 + 32;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v28 = (__int64 *)sub_22077B0(32);
      v29 = v28;
      if ( v28 )
      {
        *v28 = 0;
        v28[1] = 0;
        v28[2] = 0;
        v28[3] = 0;
      }
      v30 = sub_2207820(4096);
      v29[2] = 4096;
      *v29 = v30;
      v8 = v30;
      v31 = *(_QWORD *)(a1 + 16);
      v29[1] = 32;
      v29[3] = v31;
      *(_QWORD *)(a1 + 16) = v29;
      if ( !v8 )
        goto LABEL_41;
    }
    else
    {
      if ( !v27 )
        goto LABEL_41;
      v8 = v27;
    }
    *(_BYTE *)(v8 + 12) = 0;
    *(_DWORD *)(v8 + 8) = 15;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)v8 = &unk_49E1120;
    *(_DWORD *)(v8 + 24) = 3;
    goto LABEL_21;
  }
  if ( v4 == 84 )
  {
    v10 = *(_QWORD **)(a1 + 16);
    v11 = (*v10 + v10[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
    v10[1] = v11 - *v10 + 32;
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
    {
      v18 = (__int64 *)sub_22077B0(32);
      v19 = v18;
      if ( v18 )
      {
        *v18 = 0;
        v18[1] = 0;
        v18[2] = 0;
        v18[3] = 0;
      }
      v20 = sub_2207820(4096);
      v19[2] = 4096;
      *v19 = v20;
      v8 = v20;
      v21 = *(_QWORD *)(a1 + 16);
      v19[1] = 32;
      v19[3] = v21;
      *(_QWORD *)(a1 + 16) = v19;
      if ( !v8 )
        goto LABEL_41;
    }
    else
    {
      if ( !v11 )
        goto LABEL_41;
      v8 = v11;
    }
    *(_BYTE *)(v8 + 12) = 0;
    *(_DWORD *)(v8 + 8) = 15;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)v8 = &unk_49E1120;
    *(_DWORD *)(v8 + 24) = 2;
    goto LABEL_21;
  }
  if ( v4 != 85 )
    goto LABEL_41;
  v6 = *(_QWORD **)(a1 + 16);
  v7 = (*v6 + v6[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v6[1] = v7 - *v6 + 32;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
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
    v23[1] = 32;
    v23[3] = v25;
    *(_QWORD *)(a1 + 16) = v23;
    if ( v8 )
      goto LABEL_8;
LABEL_41:
    MEMORY[0x10] = sub_E27270(a1, a2);
    BUG();
  }
  if ( !v7 )
    goto LABEL_41;
  v8 = v7;
LABEL_8:
  *(_BYTE *)(v8 + 12) = 0;
  *(_DWORD *)(v8 + 8) = 15;
  *(_QWORD *)(v8 + 16) = 0;
  *(_QWORD *)v8 = &unk_49E1120;
  *(_DWORD *)(v8 + 24) = 1;
LABEL_21:
  *(_QWORD *)(v8 + 16) = sub_E27270(a1, a2);
  return v8;
}

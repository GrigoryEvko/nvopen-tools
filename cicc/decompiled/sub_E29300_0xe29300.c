// Function: sub_E29300
// Address: 0xe29300
//
unsigned __int64 __fastcall sub_E29300(__int64 a1, __int64 *a2, int a3)
{
  _QWORD *v4; // rdx
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  bool v7; // cc
  unsigned __int64 v8; // r15
  _QWORD *v9; // rax
  unsigned __int64 v10; // r12
  char *v11; // rax
  char v12; // cl
  __int64 v13; // rax
  _BYTE *v15; // rdx
  __int64 *v16; // rax
  __int64 *v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rax

  v4 = *(_QWORD **)(a1 + 16);
  v5 = (*v4 + v4[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = v5 - *v4 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v16 = (__int64 *)sub_22077B0(32);
    v17 = v16;
    if ( v16 )
    {
      *v16 = 0;
      v16[1] = 0;
      v16[2] = 0;
      v16[3] = 0;
    }
    v18 = sub_2207820(4096);
    v17[2] = 4096;
    *v17 = v18;
    v6 = v18;
    v19 = *(_QWORD *)(a1 + 16);
    v17[1] = 40;
    v17[3] = v19;
    *(_QWORD *)(a1 + 16) = v17;
    if ( !v6 )
    {
LABEL_3:
      v7 = a3 <= 15;
      if ( a3 != 15 )
        goto LABEL_4;
LABEL_16:
      *(_QWORD *)(v6 + 24) = 30;
      *(_QWORD *)(v6 + 32) = "`RTTI Complete Object Locator'";
      goto LABEL_7;
    }
  }
  else
  {
    v6 = 0;
    if ( !v5 )
      goto LABEL_3;
    v6 = v5;
  }
  *(_DWORD *)(v6 + 8) = 5;
  *(_QWORD *)(v6 + 16) = 0;
  *(_QWORD *)(v6 + 24) = 0;
  *(_QWORD *)v6 = &unk_49E0F88;
  *(_QWORD *)(v6 + 32) = 0;
  v7 = a3 <= 15;
  if ( a3 == 15 )
    goto LABEL_16;
LABEL_4:
  if ( v7 )
  {
    *(_QWORD *)(v6 + 24) = 9;
    if ( a3 == 1 )
      *(_QWORD *)(v6 + 32) = "`vftable'";
    else
      *(_QWORD *)(v6 + 32) = "`vbtable'";
  }
  else
  {
    *(_QWORD *)(v6 + 24) = 15;
    *(_QWORD *)(v6 + 32) = "`local vftable'";
  }
LABEL_7:
  v8 = sub_E263F0(a1, a2, v6);
  v9 = *(_QWORD **)(a1 + 16);
  v10 = (*v9 + v9[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v9[1] = v10 - *v9 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v20 = (__int64 *)sub_22077B0(32);
    v21 = v20;
    if ( v20 )
    {
      *v20 = 0;
      v20[1] = 0;
      v20[2] = 0;
      v20[3] = 0;
    }
    v22 = sub_2207820(4096);
    v21[2] = 4096;
    *v21 = v22;
    v10 = v22;
    v23 = *(_QWORD *)(a1 + 16);
    v21[1] = 40;
    v21[3] = v23;
    *(_QWORD *)(a1 + 16) = v21;
  }
  if ( !v10 )
  {
    MEMORY[0x10] = v8;
    BUG();
  }
  *(_BYTE *)(v10 + 32) = 0;
  *(_QWORD *)(v10 + 16) = 0;
  *(_QWORD *)(v10 + 16) = v8;
  *(_DWORD *)(v10 + 8) = 28;
  *(_QWORD *)v10 = &unk_49E12E0;
  *(_QWORD *)(v10 + 24) = 0;
  if ( *a2 && (v11 = (char *)a2[1], v12 = *v11, --*a2, a2[1] = (__int64)(v11 + 1), (unsigned __int8)(v12 - 54) <= 1u) )
  {
    *(_BYTE *)(v10 + 32) = sub_E22E40(a1, a2);
    v13 = *a2;
    if ( *a2 && (v15 = (_BYTE *)a2[1], *v15 == 64) )
    {
      a2[1] = (__int64)(v15 + 1);
      *a2 = v13 - 1;
    }
    else
    {
      *(_QWORD *)(v10 + 24) = sub_E27270(a1, a2);
    }
  }
  else
  {
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  return v10;
}

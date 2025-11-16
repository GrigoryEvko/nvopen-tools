// Function: sub_214D850
// Address: 0x214d850
//
__int64 __fastcall sub_214D850(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r14
  int v9; // eax
  unsigned int v10; // ebx
  unsigned int v11; // edi
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 *v16; // r12
  int v17; // eax
  unsigned int v18; // edi
  __int64 *v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rdi
  unsigned int v22; // r8d
  __int64 *v23; // rdx
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdi
  unsigned __int64 v28; // rax
  __int64 v29; // rdi
  int v30; // edx
  int v31; // edx
  int v32; // r9d
  int v33; // r10d
  int v34; // edx
  int v35; // r9d

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_41:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FC6A0C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_41;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FC6A0C);
  v7 = *(_QWORD *)(v6 + 240);
  v8 = v6;
  v9 = *(_DWORD *)(v6 + 256);
  if ( !v9 )
    return 0;
  v10 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v11 = (v9 - 1) & v10;
  v12 = (__int64 *)(v7 + 16LL * v11);
  v13 = *v12;
  if ( a2 != *v12 )
  {
    v34 = 1;
    while ( v13 != -8 )
    {
      v35 = v34 + 1;
      v11 = (v9 - 1) & (v34 + v11);
      v12 = (__int64 *)(v7 + 16LL * v11);
      v13 = *v12;
      if ( a2 == *v12 )
        goto LABEL_7;
      v34 = v35;
    }
    return 0;
  }
LABEL_7:
  v14 = v12[1];
  if ( !v14 )
    return 0;
  if ( a2 != **(_QWORD **)(v14 + 32) )
    return 0;
  v16 = *(__int64 **)(a2 + 64);
  if ( v16 == *(__int64 **)(a2 + 72) )
    return 0;
  while ( 1 )
  {
    v26 = *v16;
    if ( !v9 )
      goto LABEL_44;
    v17 = v9 - 1;
    v18 = v17 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v19 = (__int64 *)(v7 + 16LL * v18);
    v20 = *v19;
    if ( v26 == *v19 )
    {
LABEL_13:
      v21 = v19[1];
    }
    else
    {
      v31 = 1;
      while ( v20 != -8 )
      {
        v32 = v31 + 1;
        v18 = v17 & (v31 + v18);
        v19 = (__int64 *)(v7 + 16LL * v18);
        v20 = *v19;
        if ( v26 == *v19 )
          goto LABEL_13;
        v31 = v32;
      }
      v21 = 0;
    }
    v22 = v17 & v10;
    v23 = (__int64 *)(v7 + 16LL * (v17 & v10));
    v24 = *v23;
    if ( a2 == *v23 )
    {
LABEL_15:
      v25 = v23[1];
    }
    else
    {
      v30 = 1;
      while ( v24 != -8 )
      {
        v33 = v30 + 1;
        v22 = v17 & (v30 + v22);
        v23 = (__int64 *)(v7 + 16LL * v22);
        v24 = *v23;
        if ( a2 == *v23 )
          goto LABEL_15;
        v30 = v33;
      }
      v25 = 0;
    }
    if ( v21 == v25 )
    {
LABEL_44:
      v27 = *(_QWORD *)(v26 + 40);
      if ( v27 )
      {
        v28 = sub_157EBA0(v27);
        if ( *(_QWORD *)(v28 + 48) || *(__int16 *)(v28 + 18) < 0 )
        {
          v29 = sub_1625790(v28, 18);
          if ( v29 )
          {
            if ( sub_1AFD990(v29, "llvm.loop.unroll.disable", 0x18u) )
              return 1;
          }
        }
      }
    }
    if ( *(__int64 **)(a2 + 72) == ++v16 )
      return 0;
    v7 = *(_QWORD *)(v8 + 240);
    v9 = *(_DWORD *)(v8 + 256);
  }
}

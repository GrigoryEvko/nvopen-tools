// Function: sub_1041270
// Address: 0x1041270
//
bool __fastcall sub_1041270(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  bool result; // al
  __int64 v7; // r14
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  int v10; // ecx
  __int64 v11; // rdx
  int v12; // ecx
  unsigned int v13; // esi
  __int64 *v14; // rax
  __int64 v15; // rdi
  unsigned __int64 v16; // r8
  unsigned int v17; // esi
  __int64 *v18; // rax
  __int64 v19; // rdi
  int v20; // eax
  int v21; // r9d
  int v22; // eax
  int v23; // r8d

  if ( a2 == a3 )
    return 1;
  v4 = *(_QWORD *)(a1 + 128);
  if ( a3 == v4 )
    return 0;
  result = 1;
  if ( a2 == v4 )
    return result;
  v7 = *(_QWORD *)(a2 + 64);
  if ( !*(_BYTE *)(a1 + 164) )
  {
    if ( sub_C8CA60(a1 + 136, *(_QWORD *)(a2 + 64)) )
      goto LABEL_9;
    goto LABEL_18;
  }
  v8 = *(_QWORD **)(a1 + 144);
  v9 = &v8[*(unsigned int *)(a1 + 156)];
  if ( v8 == v9 )
  {
LABEL_18:
    sub_1040E50(a1, v7);
    goto LABEL_9;
  }
  while ( v7 != *v8 )
  {
    if ( v9 == ++v8 )
      goto LABEL_18;
  }
LABEL_9:
  v10 = *(_DWORD *)(a1 + 320);
  v11 = *(_QWORD *)(a1 + 304);
  if ( !v10 )
    return 0;
  v12 = v10 - 1;
  v13 = v12 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v11 + 16LL * (v12 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
  v15 = *v14;
  if ( a2 == *v14 )
  {
LABEL_11:
    v16 = v14[1];
  }
  else
  {
    v22 = 1;
    while ( v15 != -4096 )
    {
      v23 = v22 + 1;
      v13 = v12 & (v22 + v13);
      v14 = (__int64 *)(v11 + 16LL * v13);
      v15 = *v14;
      if ( a2 == *v14 )
        goto LABEL_11;
      v22 = v23;
    }
    v16 = 0;
  }
  v17 = v12 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v18 = (__int64 *)(v11 + 16LL * v17);
  v19 = *v18;
  if ( a3 != *v18 )
  {
    v20 = 1;
    while ( v19 != -4096 )
    {
      v21 = v20 + 1;
      v17 = v12 & (v20 + v17);
      v18 = (__int64 *)(v11 + 16LL * v17);
      v19 = *v18;
      if ( a3 == *v18 )
        return v18[1] > v16;
      v20 = v21;
    }
    return 0;
  }
  return v18[1] > v16;
}

// Function: sub_2673B80
// Address: 0x2673b80
//
__int64 __fastcall sub_2673B80(__int64 a1, __int64 *a2)
{
  unsigned __int8 *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int8 *v7; // rax
  int v8; // edx
  unsigned __int8 *v9; // r14
  unsigned __int8 v10; // cl
  __int64 v11; // rdi
  unsigned __int8 **v12; // rax
  unsigned __int8 **v13; // rdx
  __int64 v15; // rax
  __int64 v16; // rsi
  int v17; // eax
  int v18; // edx
  unsigned int v19; // eax
  unsigned __int8 *v20; // rcx
  int v21; // edi
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rsi
  int v25; // edx
  int v26; // ecx
  unsigned int v27; // edx
  __int64 v28; // rdi
  int v29; // r8d
  unsigned __int64 v30; // rdx
  __int64 v31; // rcx

  if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) <= 1 )
    return 0;
  v4 = sub_250CBE0(a2, (__int64)a2);
  v5 = *a2;
  v6 = *a2 & 3;
  if ( v6 == 3 )
  {
    v7 = *(unsigned __int8 **)((v5 & 0xFFFFFFFFFFFFFFFCLL) + 24);
  }
  else
  {
    if ( v6 == 2 )
      goto LABEL_6;
    v7 = (unsigned __int8 *)(v5 & 0xFFFFFFFFFFFFFFFCLL);
    if ( !v7 )
      goto LABEL_6;
    v8 = *v7;
    if ( (unsigned __int8)v8 <= 0x1Cu )
      goto LABEL_6;
    v30 = (unsigned int)(v8 - 34);
    if ( (unsigned __int8)v30 > 0x33u )
      goto LABEL_6;
    v31 = 0x8000000000041LL;
    if ( !_bittest64(&v31, v30) )
      goto LABEL_6;
  }
  if ( **((_BYTE **)v7 - 4) == 25 )
    return 0;
LABEL_6:
  v9 = sub_250CBE0(a2, (__int64)a2);
  v10 = sub_2509800(a2);
  if ( v10 > 6u || ((1LL << v10) & 0x54) == 0 || !sub_B2FC80((__int64)v9) && !(unsigned __int8)sub_B2FC00(v9) )
    goto LABEL_13;
  v11 = *(_QWORD *)(a1 + 208);
  if ( *(_BYTE *)(v11 + 276) )
  {
    v12 = *(unsigned __int8 ***)(v11 + 256);
    v13 = &v12[*(unsigned int *)(v11 + 268)];
    if ( v12 != v13 )
    {
      while ( v9 != *v12 )
      {
        if ( v13 == ++v12 )
          goto LABEL_11;
      }
      goto LABEL_13;
    }
LABEL_11:
    if ( !*(_QWORD *)(a1 + 4432)
      || !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int8 *))(a1 + 4440))(a1 + 4416, v9) )
    {
      return 0;
    }
    goto LABEL_13;
  }
  if ( !sub_C8CA60(v11 + 248, (__int64)v9) )
    goto LABEL_11;
LABEL_13:
  if ( !v4 )
    return 1;
  if ( *(_BYTE *)(a1 + 4296) )
    return 1;
  v15 = *(_QWORD *)(a1 + 200);
  if ( !*(_DWORD *)(v15 + 40) )
    return 1;
  v16 = *(_QWORD *)(v15 + 8);
  v17 = *(_DWORD *)(v15 + 24);
  if ( v17 )
  {
    v18 = v17 - 1;
    v19 = (v17 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v20 = *(unsigned __int8 **)(v16 + 8LL * v19);
    if ( v4 == v20 )
      return 1;
    v21 = 1;
    while ( v20 != (unsigned __int8 *)-4096LL )
    {
      v19 = v18 & (v21 + v19);
      v20 = *(unsigned __int8 **)(v16 + 8LL * v19);
      if ( v4 == v20 )
        return 1;
      ++v21;
    }
  }
  v22 = sub_25096F0(a2);
  v23 = *(_QWORD *)(a1 + 200);
  if ( !*(_DWORD *)(v23 + 40) )
    return 1;
  v24 = *(_QWORD *)(v23 + 8);
  v25 = *(_DWORD *)(v23 + 24);
  if ( v25 )
  {
    v26 = v25 - 1;
    v27 = (v25 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v28 = *(_QWORD *)(v24 + 8LL * v27);
    if ( v22 != v28 )
    {
      v29 = 1;
      while ( v28 != -4096 )
      {
        v27 = v26 & (v29 + v27);
        v28 = *(_QWORD *)(v24 + 8LL * v27);
        if ( v22 == v28 )
          return 1;
        ++v29;
      }
      return 0;
    }
    return 1;
  }
  return 0;
}

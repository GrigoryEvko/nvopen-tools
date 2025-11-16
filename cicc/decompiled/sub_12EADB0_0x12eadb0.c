// Function: sub_12EADB0
// Address: 0x12eadb0
//
__int64 __fastcall sub_12EADB0(__int64 a1)
{
  __int64 v1; // r10
  unsigned int v2; // esi
  __int64 v4; // r8
  unsigned int v5; // edi
  int *v6; // rax
  int v7; // ecx
  __int64 result; // rax
  int v9; // r11d
  int *v10; // rdx
  int v11; // eax
  int v12; // ecx
  int v13; // eax
  int v14; // eax
  __int64 v15; // r8
  unsigned int v16; // edi
  int v17; // esi
  int v18; // r10d
  int *v19; // r9
  int v20; // eax
  int v21; // eax
  __int64 v22; // r8
  int v23; // r10d
  unsigned int v24; // edi
  int v25; // esi
  int *v26; // r12

  v1 = a1 + 1048;
  v2 = *(_DWORD *)(a1 + 1072);
  if ( !v2 )
  {
    ++*(_QWORD *)(a1 + 1048);
    goto LABEL_15;
  }
  v4 = *(_QWORD *)(a1 + 1056);
  v5 = ((_BYTE)v2 - 1) & 0xDE;
  v6 = (int *)(v4 + 16LL * ((unsigned __int8)(v2 - 1) & 0xDE));
  v7 = *v6;
  if ( *v6 == 6 )
  {
    result = *((_QWORD *)v6 + 1);
    goto LABEL_4;
  }
  v9 = 1;
  v10 = 0;
  while ( v7 != 0x7FFFFFFF )
  {
    if ( v10 || v7 != 0x80000000 )
      v6 = v10;
    v5 = (v2 - 1) & (v9 + v5);
    v26 = (int *)(v4 + 16LL * v5);
    v7 = *v26;
    if ( *v26 == 6 )
    {
      result = *((_QWORD *)v26 + 1);
      goto LABEL_4;
    }
    ++v9;
    v10 = v6;
    v6 = (int *)(v4 + 16LL * v5);
  }
  if ( !v10 )
    v10 = v6;
  v11 = *(_DWORD *)(a1 + 1064);
  ++*(_QWORD *)(a1 + 1048);
  v12 = v11 + 1;
  if ( 4 * (v11 + 1) >= 3 * v2 )
  {
LABEL_15:
    sub_12EABE0(v1, 2 * v2);
    v13 = *(_DWORD *)(a1 + 1072);
    if ( v13 )
    {
      v14 = v13 - 1;
      v15 = *(_QWORD *)(a1 + 1056);
      v16 = v14 & 0xDE;
      v12 = *(_DWORD *)(a1 + 1064) + 1;
      v10 = (int *)(v15 + 16LL * ((unsigned __int8)v14 & 0xDE));
      v17 = *v10;
      if ( *v10 == 6 )
        goto LABEL_11;
      v18 = 1;
      v19 = 0;
      while ( v17 != 0x7FFFFFFF )
      {
        if ( !v19 && v17 == 0x80000000 )
          v19 = v10;
        v16 = v14 & (v18 + v16);
        v10 = (int *)(v15 + 16LL * v16);
        v17 = *v10;
        if ( *v10 == 6 )
          goto LABEL_11;
        ++v18;
      }
LABEL_19:
      if ( v19 )
        v10 = v19;
      goto LABEL_11;
    }
LABEL_41:
    ++*(_DWORD *)(a1 + 1064);
    BUG();
  }
  if ( v2 - *(_DWORD *)(a1 + 1068) - v12 <= v2 >> 3 )
  {
    sub_12EABE0(v1, v2);
    v20 = *(_DWORD *)(a1 + 1072);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 1056);
      v19 = 0;
      v23 = 1;
      v24 = v21 & 0xDE;
      v12 = *(_DWORD *)(a1 + 1064) + 1;
      v10 = (int *)(v22 + 16LL * ((unsigned __int8)v21 & 0xDE));
      v25 = *v10;
      if ( *v10 == 6 )
        goto LABEL_11;
      while ( v25 != 0x7FFFFFFF )
      {
        if ( !v19 && v25 == 0x80000000 )
          v19 = v10;
        v24 = v21 & (v23 + v24);
        v10 = (int *)(v22 + 16LL * v24);
        v25 = *v10;
        if ( *v10 == 6 )
          goto LABEL_11;
        ++v23;
      }
      goto LABEL_19;
    }
    goto LABEL_41;
  }
LABEL_11:
  *(_DWORD *)(a1 + 1064) = v12;
  if ( *v10 != 0x7FFFFFFF )
    --*(_DWORD *)(a1 + 1068);
  *v10 = 6;
  result = 0;
  *((_QWORD *)v10 + 1) = 0;
LABEL_4:
  *(_BYTE *)(result + 13) = *(_BYTE *)(*(_QWORD *)(a1 + 1080) + 36LL);
  return result;
}

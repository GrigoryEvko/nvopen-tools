// Function: sub_1ABB600
// Address: 0x1abb600
//
__int64 __fastcall sub_1ABB600(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v6; // rdx
  int v7; // eax
  int v8; // esi
  __int64 v9; // r8
  unsigned int v10; // eax
  __int64 v11; // rdi
  int v13; // r9d
  __int64 v14; // rbx
  __int64 v15; // r14
  int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-40h]

  v20 = sub_164A190(a2);
  v3 = *(_QWORD *)(**(_QWORD **)(a1 + 72) + 56LL);
  v4 = *(_QWORD *)(v3 + 80);
  v5 = v3 + 72;
  if ( v4 == v5 )
    return 1;
  while ( 1 )
  {
    v6 = v4 - 24;
    if ( !v4 )
      v6 = 0;
    v7 = *(_DWORD *)(a1 + 64);
    if ( !v7 )
      break;
    v8 = v7 - 1;
    v9 = *(_QWORD *)(a1 + 48);
    v10 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v11 = *(_QWORD *)(v9 + 8LL * v10);
    if ( v6 != v11 )
    {
      v13 = 1;
      while ( v11 != -8 )
      {
        v10 = v8 & (v13 + v10);
        v11 = *(_QWORD *)(v9 + 8LL * v10);
        if ( v6 == v11 )
          goto LABEL_6;
        ++v13;
      }
      break;
    }
LABEL_6:
    v4 = *(_QWORD *)(v4 + 8);
    if ( v5 == v4 )
      return 1;
  }
  v14 = *(_QWORD *)(v6 + 48);
  v15 = v6 + 40;
  if ( v14 == v6 + 40 )
    goto LABEL_6;
  while ( 1 )
  {
    if ( !v14 )
      BUG();
    v16 = *(unsigned __int8 *)(v14 - 8);
    if ( (_BYTE)v16 != 78 )
      break;
    v18 = *(_QWORD *)(v14 - 48);
    if ( *(_BYTE *)(v18 + 16) )
      goto LABEL_14;
    if ( (*(_BYTE *)(v18 + 33) & 0x20) != 0 && (unsigned int)(*(_DWORD *)(v18 + 36) - 35) <= 3 )
      goto LABEL_17;
    if ( (*(_BYTE *)(v18 + 33) & 0x20) == 0 )
    {
LABEL_14:
      if ( (unsigned __int8)sub_15F3040(v14 - 24) || sub_15F3330(v14 - 24) )
        return 0;
    }
    else if ( (unsigned int)(*(_DWORD *)(v18 + 36) - 116) > 1 )
    {
      return 0;
    }
LABEL_17:
    v14 = *(_QWORD *)(v14 + 8);
    if ( v15 == v14 )
      goto LABEL_6;
  }
  if ( (unsigned int)(v16 - 54) > 1 )
    goto LABEL_14;
  v17 = *(_QWORD *)(v14 - 48);
  if ( *(_BYTE *)(v17 + 16) <= 0x10u )
    goto LABEL_17;
  v19 = sub_164A190(v17);
  if ( *(_BYTE *)(v19 + 16) == 53 && v20 != v19 )
    goto LABEL_17;
  return 0;
}

// Function: sub_1EE76C0
// Address: 0x1ee76c0
//
char __fastcall sub_1EE76C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  bool v11; // zf
  __int64 i; // rcx
  unsigned int v13; // esi
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r9
  unsigned __int64 v18; // r12
  int v19; // edx
  int v20; // r10d

  if ( !sub_1EE6200(a1) )
    sub_1EE6470(a1, a2, v3, v4, v5, v6);
  if ( !*(_BYTE *)(a1 + 56) && sub_1EE61D0(a1) )
    sub_1EE5FD0(*(_QWORD *)(a1 + 48), *(_QWORD *)(a1 + 64));
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  v8 = **(_QWORD **)(a1 + 64) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v8 )
LABEL_35:
    BUG();
  v9 = *(_QWORD *)v8;
  if ( (*(_QWORD *)v8 & 4) == 0 && (*(_BYTE *)(v8 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v8 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
        break;
      v9 = *(_QWORD *)v8;
    }
  }
LABEL_6:
  if ( v7 != v8 )
  {
    while ( (unsigned __int16)(**(_WORD **)(v8 + 16) - 12) <= 1u )
    {
      v8 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v8 )
        goto LABEL_35;
      v10 = *(_QWORD *)v8;
      if ( (*(_QWORD *)v8 & 4) != 0 || (*(_BYTE *)(v8 + 46) & 4) == 0 )
        goto LABEL_6;
      while ( 1 )
      {
        v8 = v10 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
          break;
        v10 = *(_QWORD *)v8;
      }
      if ( v7 == v8 )
        break;
    }
  }
  v11 = *(_BYTE *)(a1 + 56) == 0;
  *(_QWORD *)(a1 + 64) = v8;
  if ( !v11 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL);
          (*(_BYTE *)(v8 + 46) & 4) != 0;
          v8 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL )
    {
      ;
    }
    v13 = *(_DWORD *)(i + 384);
    v14 = *(_QWORD *)(i + 368);
    if ( v13 )
    {
      v15 = (v13 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( v8 == *v16 )
        goto LABEL_28;
      v19 = 1;
      while ( v17 != -8 )
      {
        v20 = v19 + 1;
        v15 = (v13 - 1) & (v19 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( *v16 == v8 )
          goto LABEL_28;
        v19 = v20;
      }
    }
    v16 = (__int64 *)(v14 + 16LL * v13);
LABEL_28:
    v18 = v16[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
    LOBYTE(v8) = sub_1EE61D0(a1);
    if ( (_BYTE)v8 )
      LOBYTE(v8) = sub_1EE5F90(*(_QWORD *)(a1 + 48), v18);
  }
  return v8;
}

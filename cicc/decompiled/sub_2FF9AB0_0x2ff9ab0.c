// Function: sub_2FF9AB0
// Address: 0x2ff9ab0
//
__int64 __fastcall sub_2FF9AB0(__int64 a1, unsigned __int64 a2, __int64 *a3)
{
  unsigned __int64 v5; // rax
  __int64 i; // r9
  __int64 j; // rsi
  __int16 v8; // dx
  __int64 v9; // rsi
  __int64 v10; // r8
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r10
  __int64 v14; // rbx
  __int64 v15; // rax
  unsigned int v16; // r8d
  __int64 v17; // rax
  int v19; // edx
  int v20; // r11d

  v5 = a2;
  for ( i = *(_QWORD *)(a1 + 32); (*(_BYTE *)(v5 + 44) & 4) != 0; v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  for ( ; (*(_BYTE *)(a2 + 44) & 8) != 0; a2 = *(_QWORD *)(a2 + 8) )
    ;
  for ( j = *(_QWORD *)(a2 + 8); j != v5; v5 = *(_QWORD *)(v5 + 8) )
  {
    v8 = *(_WORD *)(v5 + 68);
    if ( (unsigned __int16)(v8 - 14) > 4u && v8 != 24 )
      break;
  }
  v9 = *(unsigned int *)(i + 144);
  v10 = *(_QWORD *)(i + 128);
  if ( (_DWORD)v9 )
  {
    v11 = (v9 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v12 = (__int64 *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( *v12 == v5 )
      goto LABEL_11;
    v19 = 1;
    while ( v13 != -4096 )
    {
      v20 = v19 + 1;
      v11 = (v9 - 1) & (v19 + v11);
      v12 = (__int64 *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( *v12 == v5 )
        goto LABEL_11;
      v19 = v20;
    }
  }
  v12 = (__int64 *)(v10 + 16 * v9);
LABEL_11:
  v14 = v12[1];
  v15 = sub_2E09D00(a3, v14);
  v16 = 0;
  v17 = *(_QWORD *)(v15 + 8);
  if ( (v17 & 6) != 0 )
    LOBYTE(v16) = (v17 & 0xFFFFFFFFFFFFFFF8LL) == (v14 & 0xFFFFFFFFFFFFFFF8LL);
  return v16;
}

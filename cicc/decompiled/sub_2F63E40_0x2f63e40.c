// Function: sub_2F63E40
// Address: 0x2f63e40
//
__int64 __fastcall sub_2F63E40(_QWORD *a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rcx
  __int64 v7; // r15
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v12; // r14
  unsigned __int16 v13; // ax
  __int64 *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rbx
  unsigned __int64 v17; // rsi
  __int64 v18; // r9
  unsigned __int64 i; // rax
  __int64 j; // r8
  __int16 v21; // dx
  unsigned int v22; // r8d
  __int64 v23; // r10
  unsigned int v24; // esi
  __int64 *v25; // rcx
  __int64 v26; // r13
  __int64 v27; // r15
  __int64 v28; // r12
  __int64 v29; // rbx
  __int64 *v30; // rax
  int v31; // ecx
  int v32; // edx

  v5 = a2;
  v6 = a1[2];
  if ( a3 < 0 )
    v7 = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 16LL * (a3 & 0x7FFFFFFF) + 8);
  else
    v7 = *(_QWORD *)(*(_QWORD *)(v6 + 304) + 8LL * (unsigned int)a3);
  while ( v7 )
  {
    if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 && (*(_BYTE *)(v7 + 4) & 8) == 0 )
    {
      v12 = v7;
LABEL_10:
      v13 = (*(_DWORD *)v12 >> 8) & 0xFFF;
      if ( !v13 || (*(_BYTE *)(v12 + 4) & 1) != 0 )
      {
LABEL_12:
        while ( 1 )
        {
          v12 = *(_QWORD *)(v12 + 32);
          if ( !v12 )
            break;
          while ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
          {
            if ( (*(_BYTE *)(v12 + 4) & 8) == 0 )
              goto LABEL_10;
            v12 = *(_QWORD *)(v12 + 32);
            if ( !v12 )
              goto LABEL_16;
          }
        }
LABEL_16:
        v5 = a2;
        break;
      }
      v14 = (__int64 *)(*(_QWORD *)(a1[3] + 272LL) + 16LL * v13);
      v15 = *v14;
      v16 = v14[1];
      v17 = *(_QWORD *)(v12 + 16);
      v18 = *(_QWORD *)(a1[5] + 32LL);
      for ( i = v17; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
        ;
      for ( ; (*(_BYTE *)(v17 + 44) & 8) != 0; v17 = *(_QWORD *)(v17 + 8) )
        ;
      for ( j = *(_QWORD *)(v17 + 8); j != i; i = *(_QWORD *)(i + 8) )
      {
        v21 = *(_WORD *)(i + 68);
        if ( (unsigned __int16)(v21 - 14) > 4u && v21 != 24 )
          break;
      }
      v22 = *(_DWORD *)(v18 + 144);
      v23 = *(_QWORD *)(v18 + 128);
      if ( v22 )
      {
        v24 = (v22 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
        v25 = (__int64 *)(v23 + 16LL * v24);
        v26 = *v25;
        if ( *v25 == i )
          goto LABEL_27;
        v31 = 1;
        while ( v26 != -4096 )
        {
          v32 = v31 + 1;
          v24 = (v22 - 1) & (v31 + v24);
          v25 = (__int64 *)(v23 + 16LL * v24);
          v26 = *v25;
          if ( *v25 == i )
            goto LABEL_27;
          v31 = v32;
        }
      }
      v25 = (__int64 *)(v23 + 16LL * v22);
LABEL_27:
      v27 = v25[1];
      v28 = *(_QWORD *)(a2 + 104);
      if ( v28 )
      {
        v29 = a4 & v15 | a5 & v16;
        while ( 1 )
        {
          v30 = (__int64 *)sub_2E09D00((__int64 *)v28, v27);
          if ( v30 == (__int64 *)(*(_QWORD *)v28 + 24LL * *(unsigned int *)(v28 + 8))
            || (*(_DWORD *)((*v30 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v30 >> 1) & 3) > (*(_DWORD *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | (unsigned int)(v27 >> 1) & 3) )
          {
            if ( v29 )
              break;
          }
          v28 = *(_QWORD *)(v28 + 104);
          if ( !v28 )
            goto LABEL_12;
        }
        *(_BYTE *)(v12 + 4) |= 1u;
      }
      goto LABEL_12;
    }
    v7 = *(_QWORD *)(v7 + 32);
  }
  sub_2E0AF60(v5);
  return sub_2E168A0((_QWORD *)a1[5], v5, 0, v8, v9, v10);
}

// Function: sub_1911DB0
// Address: 0x1911db0
//
unsigned __int64 __fastcall sub_1911DB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  char v5; // al
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // r14
  __int64 v11; // rax
  bool v12; // r13
  unsigned int v14; // esi
  int v15; // eax
  int v16; // eax
  int v17; // r8d
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  int v25; // edx
  __int64 v26; // rcx
  unsigned __int64 v27; // r8
  __int64 v28; // rdx
  __int64 v29[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a1 + 32;
  v5 = sub_190F0D0(a1 + 32, a2, v29);
  v10 = v29[0];
  if ( v5 )
  {
    v11 = *(unsigned int *)(v29[0] + 56);
    v12 = (_DWORD)v11 == 0;
    if ( (_DWORD)v11 )
      return ((unsigned __int64)v12 << 32) | v11;
    goto LABEL_9;
  }
  v14 = *(_DWORD *)(a1 + 56);
  v15 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  v16 = v15 + 1;
  v17 = 2 * v14;
  if ( 4 * v16 >= 3 * v14 )
  {
    v14 *= 2;
    goto LABEL_22;
  }
  v18 = v14 - *(_DWORD *)(a1 + 52) - v16;
  v19 = v14 >> 3;
  if ( (unsigned int)v18 <= (unsigned int)v19 )
  {
LABEL_22:
    sub_1911AB0(v2, v14);
    sub_190F0D0(v2, a2, v29);
    v10 = v29[0];
    v16 = *(_DWORD *)(a1 + 48) + 1;
  }
  *(_DWORD *)(a1 + 48) = v16;
  if ( *(_DWORD *)v10 != -1 )
    --*(_DWORD *)(a1 + 52);
  v12 = 1;
  *(_DWORD *)v10 = *(_DWORD *)a2;
  *(_QWORD *)(v10 + 8) = *(_QWORD *)(a2 + 8);
  *(_BYTE *)(v10 + 16) = *(_BYTE *)(a2 + 16);
  sub_1909410(v10 + 24, a2 + 24, v18, v19, v17, v9);
  *(_DWORD *)(v10 + 56) = 0;
LABEL_9:
  v20 = *(_QWORD *)(a1 + 80);
  if ( v20 == *(_QWORD *)(a1 + 88) )
  {
    sub_190CCE0((__int64 *)(a1 + 72), (int *)v20, a2, v7);
  }
  else
  {
    if ( v20 )
    {
      *(_DWORD *)v20 = *(_DWORD *)a2;
      *(_QWORD *)(v20 + 8) = *(_QWORD *)(a2 + 8);
      *(_BYTE *)(v20 + 16) = *(_BYTE *)(a2 + 16);
      *(_QWORD *)(v20 + 24) = v20 + 40;
      *(_QWORD *)(v20 + 32) = 0x400000000LL;
      if ( *(_DWORD *)(a2 + 32) )
        sub_1909410(v20 + 24, a2 + 24, v6, v7, v8, v9);
      v20 = *(_QWORD *)(a1 + 80);
    }
    *(_QWORD *)(a1 + 80) = v20 + 56;
  }
  v21 = *(_QWORD *)(a1 + 96);
  v22 = *(unsigned int *)(a1 + 208);
  v23 = (*(_QWORD *)(a1 + 104) - v21) >> 2;
  v24 = (unsigned int)(v22 + 1);
  if ( v24 > v23 )
  {
    v27 = (unsigned int)(2 * v22);
    if ( v27 > v23 )
    {
      sub_C17A60(a1 + 96, v27 - v23);
      v22 = *(unsigned int *)(a1 + 208);
      LODWORD(v24) = v22 + 1;
    }
    else if ( v27 < v23 )
    {
      v28 = v21 + 4 * v27;
      if ( *(_QWORD *)(a1 + 104) != v28 )
        *(_QWORD *)(a1 + 104) = v28;
    }
  }
  *(_DWORD *)(v10 + 56) = v22;
  v25 = *(_DWORD *)(a1 + 64);
  *(_DWORD *)(a1 + 208) = v24;
  v26 = *(_QWORD *)(a1 + 96);
  *(_DWORD *)(a1 + 64) = v25 + 1;
  *(_DWORD *)(v26 + 4 * v22) = v25;
  v11 = *(unsigned int *)(v10 + 56);
  return ((unsigned __int64)v12 << 32) | v11;
}

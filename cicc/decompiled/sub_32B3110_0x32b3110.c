// Function: sub_32B3110
// Address: 0x32b3110
//
__int64 __fastcall sub_32B3110(__int64 a1, unsigned int a2)
{
  __int64 v2; // r13
  unsigned __int64 *v4; // rcx
  int v5; // r12d
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v13; // rdi

  v2 = a2;
  v4 = *(unsigned __int64 **)(a1 + 144);
  v5 = *(_DWORD *)(a1 + 140);
  v6 = *v4;
  if ( *v4 )
  {
    *v4 = *(_QWORD *)v6;
LABEL_3:
    memset((void *)v6, 0, 0xC0u);
    v7 = v6 & 0xFFFFFFFFFFFFFFC0LL;
    goto LABEL_4;
  }
  v13 = v4[1];
  v4[11] += 192LL;
  v7 = (v13 + 63) & 0xFFFFFFFFFFFFFFC0LL;
  if ( v4[2] < v7 + 192 || !v13 )
  {
    v6 = sub_9D1E70((__int64)(v4 + 1), 192, 192, 6);
    goto LABEL_3;
  }
  v4[1] = v7 + 192;
  if ( v7 )
  {
    v6 = (v13 + 63) & 0xFFFFFFFFFFFFFFC0LL;
    goto LABEL_3;
  }
LABEL_4:
  v8 = 0;
  if ( v5 )
  {
    do
    {
      *(_QWORD *)(v6 + 8 * v8) = *(_QWORD *)(a1 + 8 * v8 + 8);
      *(_QWORD *)(v6 + 8 * v8 + 96) = *(_QWORD *)(a1 + 8 * v8 + 72);
      ++v8;
    }
    while ( v5 != (_DWORD)v8 );
  }
  v9 = (unsigned int)(v5 - 1);
  v10 = v9 | v7;
  v11 = *(_QWORD *)(((v9 | v7) & 0xFFFFFFFFFFFFFFC0LL) + 8 * v9 + 0x60);
  *(_QWORD *)(a1 + 8) = v10;
  ++*(_DWORD *)(a1 + 136);
  *(_QWORD *)(a1 + 72) = v11;
  *(_DWORD *)(a1 + 140) = 1;
  return v2 << 32;
}

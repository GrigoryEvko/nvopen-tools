// Function: sub_2FBA950
// Address: 0x2fba950
//
__int64 __fastcall sub_2FBA950(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  unsigned __int64 *v4; // rax
  int v5; // r13d
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r13
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v13; // rsi

  v2 = a2;
  v4 = *(unsigned __int64 **)(a1 + 192);
  v5 = *(_DWORD *)(a1 + 188);
  v6 = *v4;
  if ( *v4 )
  {
    *v4 = *(_QWORD *)v6;
LABEL_3:
    memset((void *)v6, 0, 0xC0u);
    memset(
      (void *)((v6 + 104) & 0xFFFFFFFFFFFFFFF8LL),
      0,
      8LL * (((unsigned int)v6 - (((_DWORD)v6 + 104) & 0xFFFFFFF8) + 192) >> 3));
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
  if ( v5 )
  {
    v8 = 1;
    v9 = (unsigned int)(v5 - 1);
    do
    {
      *(_QWORD *)(v6 + 8 * v8 - 8) = *(_QWORD *)(a1 + 8 * v8);
      *(_QWORD *)(v6 + 8 * v8 + 88) = *(_QWORD *)(a1 + 8 * v8 + 88);
      ++v8;
    }
    while ( v9 + 2 != v8 );
  }
  else
  {
    LODWORD(v9) = -1;
  }
  v10 = (unsigned int)v9 | v7;
  v11 = *(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFC0LL) + 8LL * (unsigned int)v9 + 0x60);
  ++*(_DWORD *)(a1 + 184);
  *(_QWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 96) = v11;
  *(_DWORD *)(a1 + 188) = 1;
  return v2 << 32;
}

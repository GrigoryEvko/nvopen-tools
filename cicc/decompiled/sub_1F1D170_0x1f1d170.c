// Function: sub_1F1D170
// Address: 0x1f1d170
//
unsigned __int64 __fastcall sub_1F1D170(__int64 a1, unsigned int a2)
{
  int v3; // r13d
  __int64 *v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r13
  unsigned __int64 v8; // rdx
  __int64 v9; // rax

  v3 = *(_DWORD *)(a1 + 188);
  v4 = *(__int64 **)(a1 + 192);
  v5 = *v4;
  if ( *v4 )
    *v4 = *(_QWORD *)v5;
  else
    v5 = sub_145CBF0(v4 + 1, 192, 64);
  memset((void *)v5, 0, 0xC0u);
  if ( v3 )
  {
    v6 = 1;
    v7 = (unsigned int)(v3 - 1);
    do
    {
      *(_QWORD *)(v5 + 8 * v6 - 8) = *(_QWORD *)(a1 + 8 * v6);
      *(_QWORD *)(v5 + 8 * v6 + 88) = *(_QWORD *)(a1 + 8 * v6 + 88);
      ++v6;
    }
    while ( v7 + 2 != v6 );
  }
  else
  {
    LODWORD(v7) = -1;
  }
  v8 = (unsigned int)v7 | v5 & 0xFFFFFFFFFFFFFFC0LL;
  v9 = *(_QWORD *)((v8 & 0xFFFFFFFFFFFFFFC0LL) + 8LL * (unsigned int)v7 + 0x60);
  ++*(_DWORD *)(a1 + 184);
  *(_QWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 96) = v9;
  *(_DWORD *)(a1 + 188) = 1;
  return (unsigned __int64)a2 << 32;
}

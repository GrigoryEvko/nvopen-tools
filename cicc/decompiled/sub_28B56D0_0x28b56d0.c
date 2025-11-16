// Function: sub_28B56D0
// Address: 0x28b56d0
//
__int64 __fastcall sub_28B56D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 *v6; // rcx
  __int64 v7; // rdi
  __int64 *v8; // rax
  unsigned int v9; // eax
  int v10; // eax
  __int64 *v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rax
  int v14; // esi
  bool v15; // cc
  __int64 result; // rax
  __int64 *v17; // rax
  __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  int v22; // edx

  v2 = a2 + 8;
  v3 = a1 + 8;
  *(_QWORD *)a1 = *(_QWORD *)a2;
  if ( (*(_BYTE *)(a1 + 16) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 24), 8LL * *(unsigned int *)(a1 + 32), 8);
  *(_QWORD *)(a1 + 16) = 1;
  v6 = (__int64 *)(a1 + 88);
  v7 = a1 + 24;
  v8 = (__int64 *)(a1 + 24);
  do
  {
    if ( v8 )
      *v8 = -4096;
    ++v8;
  }
  while ( v8 != v6 );
  v9 = *(_DWORD *)(a2 + 16) & 0xFFFFFFFE;
  *(_DWORD *)(a2 + 16) = *(_DWORD *)(a1 + 16) & 0xFFFFFFFE | *(_DWORD *)(a2 + 16) & 1;
  *(_DWORD *)(a1 + 16) = v9 | *(_DWORD *)(a1 + 16) & 1;
  v10 = *(_DWORD *)(a1 + 20);
  *(_DWORD *)(a1 + 20) = *(_DWORD *)(a2 + 20);
  *(_DWORD *)(a2 + 20) = v10;
  if ( (*(_BYTE *)(a1 + 16) & 1) == 0 )
  {
    if ( (*(_BYTE *)(a2 + 16) & 1) == 0 )
    {
      v21 = *(_QWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 24);
      v22 = *(_DWORD *)(a2 + 32);
      *(_QWORD *)(a2 + 24) = v21;
      LODWORD(v21) = *(_DWORD *)(a1 + 32);
      *(_DWORD *)(a1 + 32) = v22;
      *(_DWORD *)(a2 + 32) = v21;
      goto LABEL_13;
    }
    v11 = (__int64 *)(a1 + 24);
    v7 = a2 + 24;
    v2 = v3;
    v3 = a2 + 8;
    goto LABEL_10;
  }
  v11 = (__int64 *)(a2 + 24);
  if ( (*(_BYTE *)(a2 + 16) & 1) == 0 )
  {
LABEL_10:
    *(_BYTE *)(v2 + 8) |= 1u;
    v12 = *(_QWORD *)(v2 + 16);
    v13 = 0;
    v14 = *(_DWORD *)(v2 + 24);
    do
    {
      v11[v13] = *(_QWORD *)(v7 + v13 * 8);
      ++v13;
    }
    while ( v13 != 8 );
    *(_BYTE *)(v3 + 8) &= ~1u;
    *(_QWORD *)(v3 + 16) = v12;
    *(_DWORD *)(v3 + 24) = v14;
    goto LABEL_13;
  }
  v17 = (__int64 *)(a1 + 24);
  do
  {
    v18 = *v17;
    *v17++ = *v11;
    *v11++ = v18;
  }
  while ( v17 != v6 );
LABEL_13:
  *(_BYTE *)(a1 + 88) = *(_BYTE *)(a2 + 88);
  *(_DWORD *)(a1 + 92) = *(_DWORD *)(a2 + 92);
  if ( a1 + 96 != a2 + 96 )
  {
    v15 = *(_DWORD *)(a1 + 128) <= 0x40u;
    *(_QWORD *)(a1 + 96) = *(_QWORD *)(a2 + 96);
    *(_QWORD *)(a1 + 104) = *(_QWORD *)(a2 + 104);
    *(_DWORD *)(a1 + 112) = *(_DWORD *)(a2 + 112);
    if ( !v15 )
    {
      v20 = *(_QWORD *)(a1 + 120);
      if ( v20 )
        j_j___libc_free_0_0(v20);
    }
    *(_QWORD *)(a1 + 120) = *(_QWORD *)(a2 + 120);
    *(_DWORD *)(a1 + 128) = *(_DWORD *)(a2 + 128);
    *(_DWORD *)(a2 + 128) = 0;
  }
  if ( a2 + 136 != a1 + 136 )
  {
    v15 = *(_DWORD *)(a1 + 168) <= 0x40u;
    *(_QWORD *)(a1 + 136) = *(_QWORD *)(a2 + 136);
    *(_QWORD *)(a1 + 144) = *(_QWORD *)(a2 + 144);
    *(_DWORD *)(a1 + 152) = *(_DWORD *)(a2 + 152);
    if ( !v15 )
    {
      v19 = *(_QWORD *)(a1 + 160);
      if ( v19 )
        j_j___libc_free_0_0(v19);
    }
    *(_QWORD *)(a1 + 160) = *(_QWORD *)(a2 + 160);
    *(_DWORD *)(a1 + 168) = *(_DWORD *)(a2 + 168);
    *(_DWORD *)(a2 + 168) = 0;
  }
  *(_DWORD *)(a1 + 176) = *(_DWORD *)(a2 + 176);
  result = *(_QWORD *)(a2 + 184);
  *(_QWORD *)(a1 + 184) = result;
  return result;
}

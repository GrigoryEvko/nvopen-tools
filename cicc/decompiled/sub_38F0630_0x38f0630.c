// Function: sub_38F0630
// Address: 0x38f0630
//
void __fastcall sub_38F0630(__int64 a1)
{
  int *v2; // rdx
  int v3; // eax
  unsigned __int64 v4; // rcx
  _DWORD *v5; // rbx
  int v6; // eax
  unsigned __int64 v7; // r12
  __m128i v8; // xmm0
  bool v9; // cc
  unsigned __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  int *v13; // rax
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rcx
  _DWORD *v16; // rbx
  int v17; // eax
  unsigned __int64 v18; // r12
  __m128i v19; // xmm1
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  int *v23; // rax
  unsigned __int64 v24; // rdi
  _BYTE v25[24]; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v26; // [rsp+18h] [rbp-38h]
  unsigned int v27; // [rsp+20h] [rbp-30h]

LABEL_1:
  v2 = *(int **)(a1 + 152);
  v3 = *v2;
  if ( *v2 == 9 )
  {
LABEL_16:
    v15 = *(unsigned int *)(a1 + 160);
    *(_BYTE *)(a1 + 258) = 1;
    v16 = v2 + 10;
    v17 = v15;
    v15 *= 40LL;
    v18 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v15 - 40) >> 3);
    if ( v15 > 0x28 )
    {
      do
      {
        v19 = _mm_loadu_si128((const __m128i *)(v16 + 2));
        v9 = *(v16 - 2) <= 0x40u;
        *(v16 - 10) = *v16;
        *((__m128i *)v16 - 2) = v19;
        if ( !v9 )
        {
          v20 = *((_QWORD *)v16 - 2);
          if ( v20 )
            j_j___libc_free_0_0(v20);
        }
        v21 = *((_QWORD *)v16 + 3);
        v16 += 10;
        *((_QWORD *)v16 - 7) = v21;
        LODWORD(v21) = *(v16 - 2);
        *(v16 - 2) = 0;
        *(v16 - 12) = v21;
        --v18;
      }
      while ( v18 );
      v17 = *(_DWORD *)(a1 + 160);
      v2 = *(int **)(a1 + 152);
    }
    v22 = (unsigned int)(v17 - 1);
    *(_DWORD *)(a1 + 160) = v22;
    v23 = &v2[10 * v22];
    if ( (unsigned int)v23[8] > 0x40 )
    {
      v24 = *((_QWORD *)v23 + 3);
      if ( v24 )
        j_j___libc_free_0_0(v24);
    }
    if ( !*(_DWORD *)(a1 + 160) )
    {
      sub_392C2E0(v25, a1 + 144);
      sub_38E90E0(a1 + 152, *(_QWORD *)(a1 + 152), (unsigned __int64)v25);
      if ( v27 > 0x40 )
      {
        if ( v26 )
          j_j___libc_free_0_0(v26);
      }
    }
  }
  else
  {
    while ( v3 )
    {
      v4 = *(unsigned int *)(a1 + 160);
      *(_BYTE *)(a1 + 258) = 0;
      v5 = v2 + 10;
      v6 = v4;
      v4 *= 40LL;
      v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - 40) >> 3);
      if ( v4 > 0x28 )
      {
        do
        {
          v8 = _mm_loadu_si128((const __m128i *)(v5 + 2));
          v9 = *(v5 - 2) <= 0x40u;
          *(v5 - 10) = *v5;
          *((__m128i *)v5 - 2) = v8;
          if ( !v9 )
          {
            v10 = *((_QWORD *)v5 - 2);
            if ( v10 )
              j_j___libc_free_0_0(v10);
          }
          v11 = *((_QWORD *)v5 + 3);
          v5 += 10;
          *((_QWORD *)v5 - 7) = v11;
          LODWORD(v11) = *(v5 - 2);
          *(v5 - 2) = 0;
          *(v5 - 12) = v11;
          --v7;
        }
        while ( v7 );
        v6 = *(_DWORD *)(a1 + 160);
        v2 = *(int **)(a1 + 152);
      }
      v12 = (unsigned int)(v6 - 1);
      *(_DWORD *)(a1 + 160) = v12;
      v13 = &v2[10 * v12];
      if ( (unsigned int)v13[8] > 0x40 )
      {
        v14 = *((_QWORD *)v13 + 3);
        if ( v14 )
          j_j___libc_free_0_0(v14);
      }
      if ( *(_DWORD *)(a1 + 160) )
        goto LABEL_1;
      sub_392C2E0(v25, a1 + 144);
      sub_38E90E0(a1 + 152, *(_QWORD *)(a1 + 152), (unsigned __int64)v25);
      if ( v27 <= 0x40 || !v26 )
        goto LABEL_1;
      j_j___libc_free_0_0(v26);
      v2 = *(int **)(a1 + 152);
      v3 = *v2;
      if ( *v2 == 9 )
        goto LABEL_16;
    }
  }
}

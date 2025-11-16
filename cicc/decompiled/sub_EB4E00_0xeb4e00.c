// Function: sub_EB4E00
// Address: 0xeb4e00
//
__int64 __fastcall sub_EB4E00(__int64 a1)
{
  unsigned int *v2; // rdx
  __int64 result; // rax
  unsigned __int64 v4; // rcx
  unsigned int *v5; // rbx
  int v6; // eax
  unsigned __int64 v7; // r12
  __m128i v8; // xmm0
  bool v9; // cc
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned __int64 v18; // rcx
  unsigned int *v19; // rbx
  int v20; // eax
  unsigned __int64 v21; // r12
  __m128i v22; // xmm1
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  _BYTE v31[24]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v32; // [rsp+18h] [rbp-38h]
  unsigned int v33; // [rsp+20h] [rbp-30h]

LABEL_1:
  v2 = *(unsigned int **)(a1 + 48);
  result = *v2;
  if ( (_DWORD)result == 9 )
  {
LABEL_16:
    v18 = *(unsigned int *)(a1 + 56);
    *(_BYTE *)(a1 + 155) = 1;
    v19 = v2 + 10;
    v20 = v18;
    v18 *= 40LL;
    v21 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v18 - 40) >> 3);
    if ( v18 > 0x28 )
    {
      do
      {
        v22 = _mm_loadu_si128((const __m128i *)(v19 + 2));
        v9 = *(v19 - 2) <= 0x40;
        *(v19 - 10) = *v19;
        *((__m128i *)v19 - 2) = v22;
        if ( !v9 )
        {
          v23 = *((_QWORD *)v19 - 2);
          if ( v23 )
            j_j___libc_free_0_0(v23);
        }
        v24 = *((_QWORD *)v19 + 3);
        v19 += 10;
        *((_QWORD *)v19 - 7) = v24;
        LODWORD(v24) = *(v19 - 2);
        *(v19 - 2) = 0;
        *(v19 - 12) = v24;
        --v21;
      }
      while ( v21 );
      v20 = *(_DWORD *)(a1 + 56);
      v2 = *(unsigned int **)(a1 + 48);
    }
    v25 = (unsigned int)(v20 - 1);
    *(_DWORD *)(a1 + 56) = v25;
    v26 = &v2[10 * v25];
    if ( v26[8] > 0x40 )
    {
      v27 = *((_QWORD *)v26 + 3);
      if ( v27 )
        j_j___libc_free_0_0(v27);
    }
    result = *(unsigned int *)(a1 + 56);
    if ( !(_DWORD)result )
    {
      sub_1097F60(v31, a1 + 40);
      result = sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)v31, v28, v29, v30);
      if ( v33 > 0x40 )
      {
        if ( v32 )
          return j_j___libc_free_0_0(v32);
      }
    }
  }
  else
  {
    while ( (_DWORD)result )
    {
      v4 = *(unsigned int *)(a1 + 56);
      *(_BYTE *)(a1 + 155) = 0;
      v5 = v2 + 10;
      v6 = v4;
      v4 *= 40LL;
      v7 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v4 - 40) >> 3);
      if ( v4 > 0x28 )
      {
        do
        {
          v8 = _mm_loadu_si128((const __m128i *)(v5 + 2));
          v9 = *(v5 - 2) <= 0x40;
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
        v6 = *(_DWORD *)(a1 + 56);
        v2 = *(unsigned int **)(a1 + 48);
      }
      v12 = (unsigned int)(v6 - 1);
      *(_DWORD *)(a1 + 56) = v12;
      v13 = &v2[10 * v12];
      if ( v13[8] > 0x40 )
      {
        v14 = *((_QWORD *)v13 + 3);
        if ( v14 )
          j_j___libc_free_0_0(v14);
      }
      if ( *(_DWORD *)(a1 + 56) )
        goto LABEL_1;
      sub_1097F60(v31, a1 + 40);
      sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)v31, v15, v16, v17);
      if ( v33 <= 0x40 || !v32 )
        goto LABEL_1;
      j_j___libc_free_0_0(v32);
      v2 = *(unsigned int **)(a1 + 48);
      result = *v2;
      if ( (_DWORD)result == 9 )
        goto LABEL_16;
    }
  }
  return result;
}

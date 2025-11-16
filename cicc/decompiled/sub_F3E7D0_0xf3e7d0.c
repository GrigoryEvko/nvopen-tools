// Function: sub_F3E7D0
// Address: 0xf3e7d0
//
__int64 __fastcall sub_F3E7D0(__int64 a1, __int64 a2, const __m128i *a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __m128i *v7; // r14
  __int64 v8; // rcx
  __int64 v9; // rax
  char v10; // dl
  unsigned int v12; // eax
  int v13; // eax
  unsigned int v14; // esi
  unsigned int v15; // ecx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rdx
  __m128i *v19; // [rsp+0h] [rbp-60h] BYREF
  __m128i *v20; // [rsp+8h] [rbp-58h] BYREF
  __int64 v21[3]; // [rsp+10h] [rbp-50h] BYREF
  char v22; // [rsp+28h] [rbp-38h]
  __int64 v23; // [rsp+30h] [rbp-30h]

  if ( (unsigned __int8)sub_F38D60(a2, (__int64)a3, (__int64 *)&v19) )
  {
    if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
    {
      v5 = a2 + 16;
      v6 = 160;
    }
    else
    {
      v5 = *(_QWORD *)(a2 + 16);
      v6 = 40LL * *(unsigned int *)(a2 + 24);
    }
    v7 = v19;
    v8 = *(_QWORD *)a2;
    v9 = v6 + v5;
    v10 = 0;
    goto LABEL_5;
  }
  v12 = *(_DWORD *)(a2 + 8);
  v7 = v19;
  ++*(_QWORD *)a2;
  v20 = v7;
  v13 = (v12 >> 1) + 1;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v15 = 12;
    v14 = 4;
  }
  else
  {
    v14 = *(_DWORD *)(a2 + 24);
    v15 = 3 * v14;
  }
  if ( v15 <= 4 * v13 )
  {
    v14 *= 2;
    goto LABEL_19;
  }
  if ( v14 - (v13 + *(_DWORD *)(a2 + 12)) <= v14 >> 3 )
  {
LABEL_19:
    sub_F3E3C0(a2, v14);
    sub_F38D60(a2, (__int64)a3, (__int64 *)&v20);
    v7 = v20;
    v13 = (*(_DWORD *)(a2 + 8) >> 1) + 1;
  }
  v16 = *(_DWORD *)(a2 + 8);
  v21[0] = 0;
  v22 = 0;
  v23 = 0;
  *(_DWORD *)(a2 + 8) = v16 & 1 | (2 * v13);
  if ( !sub_F34140((__int64)v7, (__int64)v21) )
    --*(_DWORD *)(a2 + 12);
  *v7 = _mm_loadu_si128(a3);
  v7[1] = _mm_loadu_si128(a3 + 1);
  v7[2].m128i_i64[0] = a3[2].m128i_i64[0];
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v17 = a2 + 16;
    v18 = 160;
  }
  else
  {
    v17 = *(_QWORD *)(a2 + 16);
    v18 = 40LL * *(unsigned int *)(a2 + 24);
  }
  v9 = v18 + v17;
  v8 = *(_QWORD *)a2;
  v10 = 1;
LABEL_5:
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v7;
  *(_QWORD *)(a1 + 24) = v9;
  *(_QWORD *)(a1 + 8) = v8;
  *(_BYTE *)(a1 + 32) = v10;
  return a1;
}

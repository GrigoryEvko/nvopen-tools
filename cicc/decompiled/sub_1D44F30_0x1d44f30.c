// Function: sub_1D44F30
// Address: 0x1d44f30
//
__int64 __fastcall sub_1D44F30(const __m128i *a1, __int64 a2)
{
  unsigned __int16 v2; // ax
  int v3; // r14d
  char v4; // r15
  char v5; // bl
  unsigned __int8 *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r9
  int v12; // edx
  int v13; // r8d
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __m128i v22; // [rsp+0h] [rbp-60h] BYREF
  __m128i v23; // [rsp+10h] [rbp-50h]
  __m128i v24; // [rsp+20h] [rbp-40h]

  v2 = *(_WORD *)(a2 + 24) - 81;
  v3 = dword_42E76A0[v2];
  v4 = byte_42E7680[v2];
  v5 = byte_42E7660[v2];
  sub_1D44C70((__int64)a1, a2, 1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v6 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 40LL)
                         + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL));
  v10 = sub_1D29190((__int64)a1, *v6, *((_QWORD *)v6 + 1), v7, v8, v9);
  v13 = v12;
  v14 = *(_QWORD *)(a2 + 32);
  if ( v4 )
  {
    v15 = 1;
    v22 = _mm_loadu_si128((const __m128i *)(v14 + 40));
  }
  else if ( v5 )
  {
    v15 = 3;
    v22 = _mm_loadu_si128((const __m128i *)(v14 + 40));
    v23 = _mm_loadu_si128((const __m128i *)(v14 + 80));
    v24 = _mm_loadu_si128((const __m128i *)(v14 + 120));
  }
  else
  {
    v15 = 2;
    v22 = _mm_loadu_si128((const __m128i *)(v14 + 40));
    v23 = _mm_loadu_si128((const __m128i *)(v14 + 80));
  }
  v16 = sub_1D2E3C0((__int64)a1, a2, v3, v10, v13, v11, v22.m128i_i64, v15);
  if ( v16 == a2 )
  {
    *(_DWORD *)(v16 + 28) = -1;
  }
  else
  {
    sub_1D444E0((__int64)a1, a2, v16);
    sub_1D2DC70(a1, a2, v17, v18, v19, v20);
  }
  return v16;
}

// Function: sub_EAA750
// Address: 0xeaa750
//
__int64 __fastcall sub_EAA750(__int64 a1)
{
  __int64 v1; // r12
  unsigned int v2; // r13d
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r10
  int v12; // [rsp+30h] [rbp-70h] BYREF
  char v13; // [rsp+38h] [rbp-68h]
  __m128i v14; // [rsp+40h] [rbp-60h] BYREF
  char v15; // [rsp+50h] [rbp-50h]
  __m128i v16; // [rsp+60h] [rbp-40h] BYREF
  __int64 v17; // [rsp+70h] [rbp-30h]

  v1 = *(_QWORD *)(a1 + 224);
  v2 = *(unsigned __int8 *)(v1 + 1793);
  if ( !(_BYTE)v2 || *(_DWORD *)(v1 + 1796) )
    return *(unsigned __int8 *)(v1 + 1793);
  v14.m128i_i32[0] = 0;
  v4 = *(_QWORD *)(v1 + 1744);
  if ( v4 )
  {
    do
    {
      v5 = v4;
      v4 = *(_QWORD *)(v4 + 16);
    }
    while ( v4 );
    if ( v1 + 1736 != v5 && !*(_DWORD *)(v5 + 32) )
      goto LABEL_9;
  }
  else
  {
    v5 = v1 + 1736;
  }
  v16.m128i_i64[0] = (__int64)&v14;
  v6 = sub_EAA600((_QWORD *)(v1 + 1728), v5, (unsigned int **)&v16);
  v1 = *(_QWORD *)(a1 + 224);
  v5 = v6;
LABEL_9:
  v7 = *(_QWORD *)(v5 + 480);
  v8 = *(_QWORD *)(v1 + 1528);
  v9 = *(_QWORD *)(v5 + 472);
  v10 = *(_QWORD *)(v1 + 1536);
  v14 = _mm_loadu_si128((const __m128i *)(v5 + 508));
  v11 = *(_QWORD *)(a1 + 232);
  v15 = *(_BYTE *)(v5 + 524);
  v16 = _mm_loadu_si128((const __m128i *)(v5 + 528));
  v17 = *(_QWORD *)(v5 + 544);
  (*(void (__fastcall **)(int *, __int64, _QWORD, __int64, __int64, _QWORD, __int64, __int64, __int64, __int64, char, __int64, __int64, __int64))(*(_QWORD *)v11 + 656LL))(
    &v12,
    v11,
    0,
    v8,
    v10,
    0,
    v9,
    v7,
    v14.m128i_i64[0],
    v14.m128i_i64[1],
    v15,
    v16.m128i_i64[0],
    v16.m128i_i64[1],
    v17);
  if ( (v13 & 1) != 0 )
    BUG();
  *(_DWORD *)(v1 + 1796) = v12;
  return v2;
}

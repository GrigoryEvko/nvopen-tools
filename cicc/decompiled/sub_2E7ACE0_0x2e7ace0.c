// Function: sub_2E7ACE0
// Address: 0x2e7ace0
//
__int64 __fastcall sub_2E7ACE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r12d
  __int64 v6; // r8
  unsigned __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned __int64 v9; // rdx
  unsigned int v10; // r15d
  unsigned __int8 v11; // al
  __int64 v12; // r14
  unsigned __int8 v13; // bl
  __int64 result; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  unsigned int v17; // [rsp+8h] [rbp-78h]
  __int64 v18; // [rsp+8h] [rbp-78h]
  unsigned __int64 v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+20h] [rbp-60h]
  _OWORD v22[5]; // [rsp+30h] [rbp-50h] BYREF

  v5 = *(unsigned __int8 *)(a2 + 34);
  v6 = a3 + *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 )
  {
    if ( (*(_QWORD *)a2 & 4) != 0 )
    {
      v20 = a3 + *(_QWORD *)(a2 + 8);
      BYTE4(v21) = *(_BYTE *)(a2 + 20);
      v19 = v7 | 4;
      LODWORD(v21) = *(_DWORD *)(v7 + 12);
    }
    else
    {
      BYTE4(v21) = *(_BYTE *)(a2 + 20);
      v16 = *(_QWORD *)(v7 + 8);
      v19 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
      v20 = v6;
      if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
        v16 = **(_QWORD **)(v16 + 16);
      LODWORD(v21) = *(_DWORD *)(v16 + 8) >> 8;
    }
  }
  else
  {
    v8 = a3 | (1LL << v5);
    v5 = -1;
    v9 = -v8 & v8;
    if ( v9 )
    {
      _BitScanReverse64(&v9, v9);
      v5 = 63 - (v9 ^ 0x3F);
    }
    v19 = 0;
    v20 = v6;
    LODWORD(v21) = *(_DWORD *)(a2 + 16);
    BYTE4(v21) = 0;
  }
  v10 = *(unsigned __int16 *)(a2 + 32);
  v11 = *(_BYTE *)(a2 + 37);
  v12 = *(unsigned __int8 *)(a2 + 36);
  v22[0] = _mm_loadu_si128((const __m128i *)(a2 + 40));
  v22[1] = _mm_loadu_si128((const __m128i *)(a2 + 56));
  v17 = v11 & 0xF;
  v13 = v11 >> 4;
  result = sub_A777F0(0x58u, (__int64 *)(a1 + 128));
  if ( result )
  {
    v15 = v17;
    v18 = result;
    sub_2EAC3E0(result, v10, a4, v5, v22, 0, v19, v20, v21, v12, v15, v13);
    return v18;
  }
  return result;
}

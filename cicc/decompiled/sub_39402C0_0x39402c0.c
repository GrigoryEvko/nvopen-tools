// Function: sub_39402C0
// Address: 0x39402c0
//
__int64 __fastcall sub_39402C0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r8
  __int64 v3; // r9
  int v4; // eax
  __int64 v5; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned int v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+8h] [rbp-28h]
  char v11; // [rsp+10h] [rbp-20h]

  sub_3940120((__int64)&v9, a2);
  if ( (v11 & 1) != 0 )
  {
    v4 = v9;
    v5 = v10;
    if ( v9 )
      goto LABEL_3;
  }
  v7 = a2[11];
  if ( v9 >= (unsigned __int64)((a2[12] - v7) >> 4) )
  {
    v5 = sub_393D180((__int64)&v9, (__int64)a2, v9, v7, v2, v3);
    v4 = 8;
LABEL_3:
    *(_DWORD *)a1 = v4;
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_QWORD *)(a1 + 8) = v5;
    return a1;
  }
  v8 = 16LL * v9;
  *(_BYTE *)(a1 + 16) &= ~1u;
  *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(v7 + v8));
  return a1;
}

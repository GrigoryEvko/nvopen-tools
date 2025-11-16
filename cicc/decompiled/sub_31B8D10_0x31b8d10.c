// Function: sub_31B8D10
// Address: 0x31b8d10
//
__int64 __fastcall sub_31B8D10(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __m128i v5; // xmm0
  _QWORD *v7; // rcx
  _QWORD *v8; // rax
  __m128i v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+10h] [rbp-20h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 80) + 16LL) == 1
    && *(_QWORD *)a1 == *(_QWORD *)(a1 + 24)
    && *(_QWORD *)(a1 + 8) == *(_QWORD *)(a1 + 32) )
  {
    v7 = *(_QWORD **)(a1 + 72);
    v8 = (_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL);
    for ( *(_QWORD *)(a1 + 64) = v8; v7 != v8; *(_QWORD *)(a1 + 64) = v8 )
    {
      if ( *v8 != -8192 && *v8 != -4096 )
        break;
      ++v8;
    }
  }
  else
  {
    sub_318E7A0(a1);
    sub_31B8600(
      &v9,
      *(_QWORD *)(a1 + 88),
      v1,
      v2,
      v3,
      v4,
      *(_OWORD *)a1,
      *(_QWORD *)(a1 + 16),
      *(_QWORD *)(a1 + 24),
      *(_QWORD *)(a1 + 32));
    v5 = _mm_loadu_si128(&v9);
    *(_QWORD *)(a1 + 16) = v10;
    *(__m128i *)a1 = v5;
  }
  return a1;
}

// Function: sub_FD5F50
// Address: 0xfd5f50
//
__int64 __fastcall sub_FD5F50(__int64 a1, const __m128i *a2, __int64 *a3)
{
  __int64 result; // rax
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // r15
  unsigned __int8 *v10; // rsi
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  _QWORD *v14; // rdi
  int v15; // eax
  int v16; // edx
  unsigned int v17; // ecx
  int v18; // eax
  int v19; // ecx
  __int64 v20; // [rsp+8h] [rbp-78h]
  __m128i v21[3]; // [rsp+10h] [rbp-70h] BYREF
  char v22; // [rsp+40h] [rbp-40h]

  result = 1;
  if ( (*(_BYTE *)(a1 + 67) & 8) == 0 )
  {
    v6 = (__int64)(a3 + 1);
    v7 = *(_QWORD *)(a1 + 24);
    v8 = v7 + 48LL * *(unsigned int *)(a1 + 32);
    if ( v7 != v8 )
    {
      while ( 1 )
      {
        v15 = sub_CF4D50(*a3, (__int64)a2, v7, v6, 0);
        v16 = (unsigned __int8)v15;
        if ( (_BYTE)v15 )
          break;
        v7 += 48;
        if ( v8 == v7 )
          goto LABEL_3;
      }
      v17 = v15;
      v18 = v15 >> 9;
      v19 = (v17 >> 8) & 1;
      return v16 | ((unsigned __int8)v19 << 8) | (unsigned int)(v18 << 9);
    }
LABEL_3:
    v9 = *(_QWORD *)(a1 + 40);
    v20 = *(_QWORD *)(a1 + 48);
    if ( v9 == v20 )
    {
LABEL_13:
      v16 = 0;
      v18 = 0;
      LOBYTE(v19) = 0;
      return v16 | ((unsigned __int8)v19 << 8) | (unsigned int)(v18 << 9);
    }
    while ( 1 )
    {
      v10 = *(unsigned __int8 **)(v9 + 16);
      v11 = _mm_loadu_si128(a2);
      v12 = _mm_loadu_si128(a2 + 1);
      v13 = _mm_loadu_si128(a2 + 2);
      v22 = 1;
      v14 = (_QWORD *)*a3;
      v21[0] = v11;
      v21[1] = v12;
      v21[2] = v13;
      if ( (unsigned __int8)sub_CF63E0(v14, v10, v21, (__int64)(a3 + 1)) )
        return 1;
      v9 += 24;
      if ( v20 == v9 )
        goto LABEL_13;
    }
  }
  return result;
}

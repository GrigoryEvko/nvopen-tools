// Function: sub_26D4D40
// Address: 0x26d4d40
//
unsigned __int64 __fastcall sub_26D4D40(int *a1, __int64 a2)
{
  __int64 v2; // r13
  int v4; // eax
  unsigned int v5; // esi
  int v6; // r12d
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 v9; // r10
  int v10; // r14d
  unsigned int v11; // ecx
  __int64 v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rsi
  unsigned __int64 result; // rax
  __int64 v16; // rcx
  int v17; // edx
  __m128i *v18; // rsi
  int v19; // eax
  int v20; // edx
  __int64 v21; // [rsp+8h] [rbp-48h] BYREF
  __m128i v22; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v23; // [rsp+20h] [rbp-30h]

  v2 = (__int64)(a1 + 2);
  v4 = *a1;
  v21 = a2;
  v5 = a1[8];
  v6 = v4 + 1;
  *a1 = v4 + 1;
  if ( !v5 )
  {
    ++*((_QWORD *)a1 + 1);
    v22.m128i_i64[0] = 0;
    goto LABEL_27;
  }
  v7 = *((_QWORD *)a1 + 2);
  v8 = v21;
  v9 = 0;
  v10 = 1;
  v11 = (v5 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v12 = v7 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( v21 != *(_QWORD *)v12 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v9 )
        v9 = v12;
      v11 = (v5 - 1) & (v10 + v11);
      v12 = v7 + 16LL * v11;
      v13 = *(_QWORD *)v12;
      if ( v21 == *(_QWORD *)v12 )
        goto LABEL_3;
      ++v10;
    }
    if ( !v9 )
      v9 = v12;
    v19 = a1[6];
    ++*((_QWORD *)a1 + 1);
    v20 = v19 + 1;
    v22.m128i_i64[0] = v9;
    if ( 4 * (v19 + 1) < 3 * v5 )
    {
      if ( v5 - a1[7] - v20 > v5 >> 3 )
        goto LABEL_21;
      goto LABEL_28;
    }
LABEL_27:
    v5 *= 2;
LABEL_28:
    sub_26D4B60(v2, v5);
    sub_26C98D0(v2, &v21, &v22);
    v8 = v21;
    v9 = v22.m128i_i64[0];
    v20 = a1[6] + 1;
LABEL_21:
    a1[6] = v20;
    if ( *(_QWORD *)v9 != -4096 )
      --a1[7];
    *(_QWORD *)v9 = v8;
    *(_DWORD *)(v9 + 8) = 0;
    *(_DWORD *)(v9 + 8) = v6;
    v14 = (_BYTE *)*((_QWORD *)a1 + 6);
    if ( v14 != *((_BYTE **)a1 + 7) )
      goto LABEL_4;
LABEL_24:
    sub_26C7040((__int64)(a1 + 10), v14, &v21);
    result = v21;
    goto LABEL_7;
  }
LABEL_3:
  *(_DWORD *)(v12 + 8) = v6;
  v14 = (_BYTE *)*((_QWORD *)a1 + 6);
  if ( v14 == *((_BYTE **)a1 + 7) )
    goto LABEL_24;
LABEL_4:
  result = v21;
  if ( v14 )
  {
    *(_QWORD *)v14 = v21;
    v14 = (_BYTE *)*((_QWORD *)a1 + 6);
  }
  *((_QWORD *)a1 + 6) = v14 + 8;
LABEL_7:
  v16 = *(_QWORD *)(result + 40);
  v17 = *a1;
  v22.m128i_i64[0] = result;
  v18 = (__m128i *)*((_QWORD *)a1 + 12);
  v22.m128i_i64[1] = v16;
  LODWORD(v23) = v17;
  if ( v18 == *((__m128i **)a1 + 13) )
    return sub_26D39A0((unsigned __int64 *)a1 + 11, v18, &v22);
  if ( v18 )
  {
    *v18 = _mm_loadu_si128(&v22);
    result = v23;
    v18[1].m128i_i64[0] = v23;
    v18 = (__m128i *)*((_QWORD *)a1 + 12);
  }
  *((_QWORD *)a1 + 12) = (char *)v18 + 24;
  return result;
}

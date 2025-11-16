// Function: sub_10FD370
// Address: 0x10fd370
//
bool __fastcall sub_10FD370(char *a1, const __m128i *a2)
{
  __int64 v2; // r13
  char v3; // r15
  __int64 v4; // r14
  int v5; // eax
  bool result; // al
  int v7; // r8d
  char v8; // al
  __m128i v9; // xmm1
  __int64 v10; // rax
  unsigned __int64 v11; // xmm2_8
  __m128i v12; // xmm3
  int v13; // ebx
  unsigned __int64 v14; // rax
  int v17; // eax
  __int64 v18; // rdi
  int v19; // ebx
  __int64 v20; // rax
  int v21; // eax
  int v22; // [rsp+Ch] [rbp-A4h]
  int v23; // [rsp+Ch] [rbp-A4h]
  bool v24; // [rsp+Ch] [rbp-A4h]
  bool v25; // [rsp+Ch] [rbp-A4h]
  int v26; // [rsp+Ch] [rbp-A4h]
  __int64 v27; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v28; // [rsp+18h] [rbp-98h]
  __int64 v29; // [rsp+20h] [rbp-90h]
  unsigned int v30; // [rsp+28h] [rbp-88h]
  __m128i v31[2]; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v32; // [rsp+50h] [rbp-60h]
  char *v33; // [rsp+58h] [rbp-58h]
  __m128i v34; // [rsp+60h] [rbp-50h]
  __int64 v35; // [rsp+70h] [rbp-40h]

  v2 = *((_QWORD *)a1 - 4);
  v3 = *a1;
  v4 = *(_QWORD *)(v2 + 8);
  v22 = sub_BCB060(v4) - (v3 == 73);
  v5 = sub_BCB090(*((_QWORD *)a1 + 1));
  if ( v22 <= v5 )
    return 1;
  v7 = v5;
  v8 = *(_BYTE *)v2;
  if ( *(_BYTE *)v2 > 0x1Cu && (v8 == 71 || v8 == 70) )
  {
    v20 = *(_QWORD *)(v2 - 32);
    if ( v20 )
    {
      v26 = v7;
      v21 = sub_BCB090(*(_QWORD *)(v20 + 8));
      v7 = v26;
      if ( v3 != 73 )
        v21 += *(_BYTE *)v2 == 71;
      if ( v21 <= v26 && v21 > 0 && v26 > 0 )
        return 1;
    }
  }
  v9 = _mm_loadu_si128(a2 + 7);
  v10 = a2[10].m128i_i64[0];
  v23 = v7;
  v11 = _mm_loadu_si128(a2 + 8).m128i_u64[0];
  v31[0] = _mm_loadu_si128(a2 + 6);
  v12 = _mm_loadu_si128(a2 + 9);
  v35 = v10;
  v32 = v11;
  v31[1] = v9;
  v33 = a1;
  v34 = v12;
  sub_9AC330((__int64)&v27, v2, 0, v31);
  v13 = sub_BCB060(v4);
  if ( v28 > 0x40 )
  {
    v19 = v13 - sub_C44500((__int64)&v27);
    result = v23 >= (int)(v19 - sub_C445E0((__int64)&v27));
    if ( v30 <= 0x40 || (v18 = v29) == 0 )
    {
LABEL_19:
      if ( v27 )
      {
        v25 = result;
        j_j___libc_free_0_0(v27);
        return v25;
      }
      return result;
    }
LABEL_18:
    v24 = result;
    j_j___libc_free_0_0(v18);
    result = v24;
    if ( v28 <= 0x40 )
      return result;
    goto LABEL_19;
  }
  if ( v28 )
  {
    if ( v27 << (64 - (unsigned __int8)v28) == -1 )
    {
      v13 -= 64;
    }
    else
    {
      _BitScanReverse64(&v14, ~(v27 << (64 - (unsigned __int8)v28)));
      v13 -= v14 ^ 0x3F;
    }
  }
  _RAX = ~v27;
  __asm { tzcnt   rdx, rax }
  v17 = 64;
  if ( v27 != -1 )
    v17 = _RDX;
  result = v23 >= v13 - v17;
  if ( v30 > 0x40 )
  {
    v18 = v29;
    if ( v29 )
      goto LABEL_18;
  }
  return result;
}

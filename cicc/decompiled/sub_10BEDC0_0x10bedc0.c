// Function: sub_10BEDC0
// Address: 0x10bedc0
//
__int64 __fastcall sub_10BEDC0(const __m128i *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r15
  int v8; // edi
  bool v9; // al
  unsigned int v10; // edi
  _BYTE *v11; // r15
  _BYTE *v12; // rdx
  _BYTE *v13; // rbx
  __int64 result; // rax
  _BYTE *v15; // rax
  int v16; // r9d
  __int64 v17; // rax
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __m128i v20; // xmm3
  unsigned int v21; // r9d
  __int64 v22; // rcx
  unsigned int **v23; // rdi
  unsigned int v24; // edx
  bool v25; // al
  int v26; // [rsp+8h] [rbp-A8h]
  int v27; // [rsp+8h] [rbp-A8h]
  __int64 v28; // [rsp+8h] [rbp-A8h]
  __int64 v29; // [rsp+8h] [rbp-A8h]
  int v30; // [rsp+8h] [rbp-A8h]
  __int64 v31; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v32; // [rsp+18h] [rbp-98h]
  __int64 v33; // [rsp+20h] [rbp-90h]
  unsigned int v34; // [rsp+28h] [rbp-88h]
  __m128i v35[2]; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v36; // [rsp+50h] [rbp-60h]
  __int64 v37; // [rsp+58h] [rbp-58h]
  __m128i v38; // [rsp+60h] [rbp-50h]
  __int64 v39; // [rsp+70h] [rbp-40h]

  v4 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v4 != 17 )
    return 0;
  v8 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( a4 )
  {
    v8 = sub_B52870(v8);
    if ( v8 != 38 )
      goto LABEL_4;
  }
  else if ( v8 != 38 )
  {
LABEL_4:
    if ( v8 == 39 )
    {
      if ( *(_DWORD *)(v4 + 32) <= 0x40u )
      {
        v9 = *(_QWORD *)(v4 + 24) == 0;
      }
      else
      {
        v26 = *(_DWORD *)(v4 + 32);
        v9 = v26 == (unsigned int)sub_C444A0(v4 + 24);
      }
      if ( v9 )
        goto LABEL_8;
    }
    return 0;
  }
  v24 = *(_DWORD *)(v4 + 32);
  if ( v24 )
  {
    if ( v24 <= 0x40 )
    {
      v25 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v24) == *(_QWORD *)(v4 + 24);
    }
    else
    {
      v30 = *(_DWORD *)(v4 + 32);
      v25 = v30 == (unsigned int)sub_C445E0(v4 + 24);
    }
    if ( !v25 )
      return 0;
  }
LABEL_8:
  v10 = *(_WORD *)(a3 + 2) & 0x3F;
  if ( a4 )
    v10 = sub_B52870(v10);
  v11 = *(_BYTE **)(a3 - 64);
  v12 = *(_BYTE **)(a2 - 64);
  v13 = *(_BYTE **)(a3 - 32);
  if ( (*v11 != 69 || v12 != *((_BYTE **)v11 - 4)) && v11 != v12 )
  {
    if ( *v13 != 69 || v12 != *((_BYTE **)v13 - 4) )
    {
      result = 0;
      if ( v13 != v12 )
        return result;
    }
    v10 = sub_B52F50(v10);
    v15 = v11;
    v11 = v13;
    v13 = v15;
  }
  if ( v10 == 40 )
  {
    v16 = 36;
  }
  else
  {
    result = 0;
    v16 = 37;
    if ( v10 != 41 )
      return result;
  }
  v17 = a1[10].m128i_i64[0];
  v18 = _mm_loadu_si128(a1 + 6);
  v19 = _mm_loadu_si128(a1 + 7);
  v27 = v16;
  v20 = _mm_loadu_si128(a1 + 9);
  v36 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v37 = a3;
  v39 = v17;
  v35[0] = v18;
  v35[1] = v19;
  v38 = v20;
  sub_9AC330((__int64)&v31, (__int64)v13, 0, v35);
  v21 = v27;
  if ( v32 > 0x40 )
    v22 = *(_QWORD *)(v31 + 8LL * ((v32 - 1) >> 6));
  else
    v22 = v31;
  result = 0;
  if ( (v22 & (1LL << ((unsigned __int8)v32 - 1))) != 0 )
  {
    if ( a4 )
      v21 = sub_B52870(v27);
    v23 = (unsigned int **)a1[2].m128i_i64[0];
    LOWORD(v36) = 257;
    result = sub_92B530(v23, v21, (__int64)v11, v13, (__int64)v35);
  }
  if ( v34 > 0x40 && v33 )
  {
    v28 = result;
    j_j___libc_free_0_0(v33);
    result = v28;
  }
  if ( v32 > 0x40 )
  {
    if ( v31 )
    {
      v29 = result;
      j_j___libc_free_0_0(v31);
      return v29;
    }
  }
  return result;
}

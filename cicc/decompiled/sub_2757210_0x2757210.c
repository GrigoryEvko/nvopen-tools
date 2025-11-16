// Function: sub_2757210
// Address: 0x2757210
//
bool __fastcall sub_2757210(__int64 a1, unsigned __int8 **a2, __int64 a3, unsigned __int64 a4)
{
  bool result; // al
  unsigned __int8 *v9; // r15
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rax
  int v13; // esi
  __int64 v14; // r8
  int v15; // esi
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r9
  __int64 v19; // rcx
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // r9
  unsigned __int8 *v23; // rax
  __int64 v24; // rdi
  int v25; // eax
  int v26; // r10d
  int v27; // eax
  int v28; // r10d
  _OWORD v29[3]; // [rsp+0h] [rbp-100h] BYREF
  __int64 v30[6]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v31[6]; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v32; // [rsp+90h] [rbp-70h] BYREF
  __m128i v33; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v34; // [rsp+B0h] [rbp-50h] BYREF
  char v35; // [rsp+C0h] [rbp-40h]
  char v36; // [rsp+C8h] [rbp-38h]

  sub_2753860((__int64)&v32, a1, (unsigned __int8 *)a4);
  if ( !v36 )
    return 0;
  v9 = sub_98ACB0(*a2, 6u);
  if ( v9 != sub_98ACB0((unsigned __int8 *)v32.m128i_i64[0], 6u) )
    return 0;
  v29[0] = _mm_loadu_si128(&v32);
  v29[1] = _mm_loadu_si128(&v33);
  v29[2] = _mm_loadu_si128(&v34);
  if ( v35 )
  {
    v23 = sub_98ACB0(*a2, 6u);
    v24 = *(_QWORD *)(a1 + 104);
    v31[0] = (__int64)v23;
    v31[1] = 1;
    memset(&v31[2], 0, 32);
    v30[0] = *(_QWORD *)&v29[0];
    v30[1] = 1;
    memset(&v30[2], 0, 32);
    return (unsigned __int8)sub_CF4D50(v24, (__int64)v30, (__int64)v31, a1 + 112, 0) == 3;
  }
  v10 = *(_QWORD *)(a3 + 40);
  v11 = *(_QWORD *)(a4 + 40);
  v30[0] = 0;
  v31[0] = 0;
  if ( v10 == v11 )
    return (unsigned int)sub_27567E0((__int64 *)a1, a4, a3, (__int64)v29, (__int64)a2, v30, v31) == 1;
  v12 = *(_QWORD *)(a1 + 824);
  v13 = *(_DWORD *)(v12 + 24);
  v14 = *(_QWORD *)(v12 + 8);
  if ( !v13 )
    goto LABEL_13;
  v15 = v13 - 1;
  v16 = v15 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v17 = (__int64 *)(v14 + 16LL * v16);
  v18 = *v17;
  if ( v10 == *v17 )
  {
LABEL_9:
    v19 = v17[1];
    if ( v19 && !*(_BYTE *)(a1 + 832) )
    {
      v20 = v15 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v21 = (__int64 *)(v14 + 16LL * v20);
      v22 = *v21;
      if ( v11 == *v21 )
      {
LABEL_12:
        if ( v19 != v21[1] )
          goto LABEL_13;
        return (unsigned int)sub_27567E0((__int64 *)a1, a4, a3, (__int64)v29, (__int64)a2, v30, v31) == 1;
      }
      v27 = 1;
      while ( v22 != -4096 )
      {
        v28 = v27 + 1;
        v20 = v15 & (v27 + v20);
        v21 = (__int64 *)(v14 + 16LL * v20);
        v22 = *v21;
        if ( v11 == *v21 )
          goto LABEL_12;
        v27 = v28;
      }
    }
  }
  else
  {
    v25 = 1;
    while ( v18 != -4096 )
    {
      v26 = v25 + 1;
      v16 = v15 & (v25 + v16);
      v17 = (__int64 *)(v14 + 16LL * v16);
      v18 = *v17;
      if ( v10 == *v17 )
        goto LABEL_9;
      v25 = v26;
    }
  }
LABEL_13:
  result = sub_2753D10(a1, *a2);
  if ( result )
    return (unsigned int)sub_27567E0((__int64 *)a1, a4, a3, (__int64)v29, (__int64)a2, v30, v31) == 1;
  return result;
}

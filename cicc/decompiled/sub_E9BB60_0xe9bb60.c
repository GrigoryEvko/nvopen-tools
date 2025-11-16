// Function: sub_E9BB60
// Address: 0xe9bb60
//
unsigned __int64 __fastcall sub_E9BB60(_QWORD *a1, unsigned int a2, unsigned int a3, _QWORD *a4)
{
  unsigned __int64 result; // rax
  unsigned __int64 v7; // rbx
  __int64 v8; // r12
  __int64 (*v9)(); // rax
  unsigned int v10; // eax
  __m128i *v11; // rsi
  __int64 v12; // rdi
  const char *v13; // rax
  __m128i v14; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v15; // [rsp+10h] [rbp-50h]
  char v16; // [rsp+20h] [rbp-40h]
  char v17; // [rsp+21h] [rbp-3Fh]

  result = sub_E99590((__int64)a1, a4);
  if ( !result )
    return result;
  v7 = result;
  if ( *(int *)(result + 76) >= 0 )
  {
    v17 = 1;
    v12 = a1[1];
    v13 = "frame register and offset can be set at most once";
LABEL_13:
    v14.m128i_i64[0] = (__int64)v13;
    v16 = 3;
    return sub_E66880(v12, a4, (__int64)&v14);
  }
  if ( (a3 & 0xF) != 0 )
  {
    v17 = 1;
    v12 = a1[1];
    v13 = "offset is not a multiple of 16";
    goto LABEL_13;
  }
  if ( a3 > 0xF0 )
  {
    v17 = 1;
    v12 = a1[1];
    v13 = "frame offset must be less than or equal to 240";
    goto LABEL_13;
  }
  v8 = 1;
  v9 = *(__int64 (**)())(*a1 + 88LL);
  if ( v9 != sub_E97650 )
    v8 = ((__int64 (__fastcall *)(_QWORD *))v9)(a1);
  v10 = sub_E91EA0(*(_QWORD *)(a1[1] + 160LL), a2);
  v14.m128i_i64[0] = v8;
  v14.m128i_i64[1] = __PAIR64__(v10, a3);
  LODWORD(v15) = 3;
  v11 = *(__m128i **)(v7 + 96);
  result = 0xAAAAAAAAAAAAAAABLL * (((__int64)v11->m128i_i64 - *(_QWORD *)(v7 + 88)) >> 3);
  *(_DWORD *)(v7 + 76) = result;
  if ( v11 == *(__m128i **)(v7 + 104) )
    return sub_E9B9B0((const __m128i **)(v7 + 88), v11, &v14);
  if ( v11 )
  {
    *v11 = _mm_loadu_si128(&v14);
    result = v15;
    v11[1].m128i_i64[0] = v15;
    v11 = *(__m128i **)(v7 + 96);
  }
  *(_QWORD *)(v7 + 96) = (char *)v11 + 24;
  return result;
}

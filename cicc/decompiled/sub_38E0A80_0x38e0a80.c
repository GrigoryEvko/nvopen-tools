// Function: sub_38E0A80
// Address: 0x38e0a80
//
unsigned __int64 __fastcall sub_38E0A80(_QWORD *a1, unsigned int a2, unsigned int a3, unsigned __int64 a4)
{
  unsigned __int64 result; // rax
  unsigned __int64 v7; // rbx
  __int64 (*v8)(); // rdx
  __int64 v9; // rax
  __m128i *v10; // rsi
  __int64 v11; // rdi
  const char *v12; // rax
  __m128i v13; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v14; // [rsp+10h] [rbp-40h]

  result = sub_38DD280((__int64)a1, a4);
  if ( !result )
    return result;
  v7 = result;
  if ( *(int *)(result + 60) >= 0 )
  {
    BYTE1(v14) = 1;
    v11 = a1[1];
    v12 = "frame register and offset can be set at most once";
LABEL_13:
    v13.m128i_i64[0] = (__int64)v12;
    LOBYTE(v14) = 3;
    return (unsigned __int64)sub_38BE3D0(v11, a4, (__int64)&v13);
  }
  if ( (a3 & 0xF) != 0 )
  {
    BYTE1(v14) = 1;
    v11 = a1[1];
    v12 = "offset is not a multiple of 16";
    goto LABEL_13;
  }
  if ( a3 > 0xF0 )
  {
    BYTE1(v14) = 1;
    v11 = a1[1];
    v12 = "frame offset must be less than or equal to 240";
    goto LABEL_13;
  }
  v8 = *(__int64 (**)())(*a1 + 16LL);
  v9 = 1;
  if ( v8 != sub_38DBC10 )
    v9 = ((__int64 (__fastcall *)(_QWORD *))v8)(a1);
  v13.m128i_i64[0] = v9;
  v13.m128i_i64[1] = __PAIR64__(a2, a3);
  LODWORD(v14) = 3;
  v10 = *(__m128i **)(v7 + 80);
  result = 0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - *(_QWORD *)(v7 + 72)) >> 3);
  *(_DWORD *)(v7 + 60) = result;
  if ( v10 == *(__m128i **)(v7 + 88) )
    return sub_38E08D0((unsigned __int64 *)(v7 + 72), v10, &v13);
  if ( v10 )
  {
    *v10 = _mm_loadu_si128(&v13);
    result = v14;
    v10[1].m128i_i64[0] = v14;
    v10 = *(__m128i **)(v7 + 80);
  }
  *(_QWORD *)(v7 + 80) = (char *)v10 + 24;
  return result;
}

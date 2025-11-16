// Function: sub_38F3F90
// Address: 0x38f3f90
//
__int64 __fastcall sub_38F3F90(__int64 a1)
{
  _DWORD *v2; // rax
  __int64 v3; // r15
  const __m128i *v4; // rsi
  _DWORD *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // ecx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // ecx
  __int64 v13; // rdi
  __int64 v14; // rax
  unsigned int v15; // r13d
  __int64 v17; // [rsp+10h] [rbp-B0h]
  __int64 v18; // [rsp+18h] [rbp-A8h]
  __int64 v19[2]; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v20; // [rsp+30h] [rbp-90h] BYREF
  __m128i *v21; // [rsp+38h] [rbp-88h]
  const __m128i *v22; // [rsp+40h] [rbp-80h]
  __int64 v23[2]; // [rsp+50h] [rbp-70h] BYREF
  char v24; // [rsp+60h] [rbp-60h]
  char v25; // [rsp+61h] [rbp-5Fh]
  __m128i v26; // [rsp+70h] [rbp-50h] BYREF
  _WORD v27[32]; // [rsp+80h] [rbp-40h] BYREF

  v2 = *(_DWORD **)(a1 + 152);
  v20 = 0;
  v21 = 0;
  v22 = 0;
  if ( *v2 == 2 )
  {
    v3 = a1 + 144;
    while ( 1 )
    {
      v6 = sub_3909290(v3);
      v19[0] = 0;
      v18 = v6;
      v19[1] = 0;
      if ( (unsigned __int8)sub_38F0EE0(a1, v19, v7, v8) )
        break;
      v9 = *(_QWORD *)(a1 + 320);
      v26.m128i_i64[0] = (__int64)v19;
      v27[0] = 261;
      v17 = sub_38BF510(v9, (__int64)&v26);
      v10 = sub_3909290(v3);
      v23[0] = 0;
      v18 = v10;
      v23[1] = 0;
      if ( (unsigned __int8)sub_38F0EE0(a1, v23, v11, v12) )
        break;
      v13 = *(_QWORD *)(a1 + 320);
      v26.m128i_i64[0] = (__int64)v23;
      v27[0] = 261;
      v14 = sub_38BF510(v13, (__int64)&v26);
      v4 = v21;
      v26.m128i_i64[1] = v14;
      v26.m128i_i64[0] = v17;
      if ( v21 == v22 )
      {
        sub_38EA190(&v20, v21, &v26);
        if ( **(_DWORD **)(a1 + 152) != 2 )
          goto LABEL_10;
      }
      else
      {
        if ( v21 )
        {
          *v21 = _mm_loadu_si128(&v26);
          v4 = v21;
        }
        v5 = *(_DWORD **)(a1 + 152);
        v21 = (__m128i *)&v4[1];
        if ( *v5 != 2 )
          goto LABEL_10;
      }
    }
    v26.m128i_i64[0] = (__int64)"expected identifier in directive";
    v27[0] = 259;
    v15 = sub_3909790(a1, v18, &v26, 0, 0);
  }
  else
  {
LABEL_10:
    LOBYTE(v27[0]) = 0;
    v26 = (__m128i)(unsigned __int64)v27;
    v25 = 1;
    v23[0] = (__int64)"unexpected token in directive";
    v24 = 3;
    if ( (unsigned __int8)sub_3909E20(a1, 25, v23) || (v15 = sub_38ECF20(a1, (unsigned __int64 *)&v26), (_BYTE)v15) )
      v15 = 1;
    else
      (*(void (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64, __int64))(**(_QWORD **)(a1 + 328) + 648LL))(
        *(_QWORD *)(a1 + 328),
        v20,
        (__int64)((__int64)v21->m128i_i64 - v20) >> 4,
        v26.m128i_i64[0],
        v26.m128i_i64[1]);
    if ( (_WORD *)v26.m128i_i64[0] != v27 )
      j_j___libc_free_0(v26.m128i_u64[0]);
  }
  if ( v20 )
    j_j___libc_free_0(v20);
  return v15;
}

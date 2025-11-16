// Function: sub_C29BE0
// Address: 0xc29be0
//
__int64 __fastcall sub_C29BE0(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned __int64 v5; // rdx
  __m128i **v6; // rax
  __m128i *v7; // rbx
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rdi
  __int64 result; // rax
  __m128i *v11; // rax
  _QWORD *v12; // r13
  _QWORD *v13; // r15
  _QWORD *v14; // rdi
  unsigned __int64 v15; // [rsp+8h] [rbp-158h] BYREF
  unsigned __int64 *v16; // [rsp+10h] [rbp-150h] BYREF
  const __m128i *v17; // [rsp+18h] [rbp-148h] BYREF
  unsigned __int64 v18; // [rsp+20h] [rbp-140h] BYREF
  char v19; // [rsp+30h] [rbp-130h]
  __m128i v20; // [rsp+40h] [rbp-120h] BYREF
  __m128i v21; // [rsp+50h] [rbp-110h] BYREF
  __int64 v22; // [rsp+60h] [rbp-100h]
  unsigned __int64 v23; // [rsp+68h] [rbp-F8h]
  char v24; // [rsp+70h] [rbp-F0h]
  _QWORD *v25[28]; // [rsp+80h] [rbp-E0h] BYREF

  *(_QWORD *)(a1 + 208) = a2;
  sub_C21E40((__int64)&v18, (_QWORD *)a1);
  if ( (v19 & 1) == 0 || (result = (unsigned int)v18, !(_DWORD)v18) )
  {
    sub_C22680((__int64)&v20, a1);
    if ( (v24 & 1) == 0 || (result = v20.m128i_u32[0], !v20.m128i_i32[0]) )
    {
      memset(v25, 0, 0xB0u);
      v25[12] = &v25[10];
      v25[13] = &v25[10];
      v25[18] = &v25[16];
      v25[19] = &v25[16];
      v5 = v23 % a3[1];
      v15 = v23;
      v6 = (__m128i **)sub_C1DD00(a3, v5, &v15, v23);
      if ( !v6 || (v7 = *v6) == 0 )
      {
        v17 = (const __m128i *)v25;
        v16 = &v15;
        v11 = sub_C286D0(a3, (__int64 **)&v16, &v17);
        v12 = v25[17];
        v7 = v11;
        while ( v12 )
        {
          v13 = v12;
          sub_C1F230((_QWORD *)v12[3]);
          v14 = (_QWORD *)v12[7];
          v12 = (_QWORD *)v12[2];
          sub_C1F480(v14);
          j_j___libc_free_0(v13, 88);
        }
      }
      sub_C1EF60(v25[11]);
      v8 = v7[5].m128i_u64[0];
      v9 = v18;
      v7[2] = _mm_loadu_si128(&v20);
      v7[3] = _mm_loadu_si128(&v21);
      v7[4].m128i_i64[0] = v22;
      v7[5].m128i_i64[0] = sub_C1B1E0(v9, 1u, v8, (bool *)v25);
      if ( (_DWORD)v22 )
        ++*(_DWORD *)(a1 + 180);
      result = sub_C27E00(a1, (unsigned __int64 *)&v7[1]);
      if ( !(_DWORD)result )
      {
        sub_C1AFD0();
        return 0;
      }
    }
  }
  return result;
}

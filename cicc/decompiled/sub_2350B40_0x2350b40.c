// Function: sub_2350B40
// Address: 0x2350b40
//
__int64 __fastcall sub_2350B40(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rcx
  __int64 v6; // rsi
  unsigned __int64 v7; // rdi
  __int64 v8; // rcx
  unsigned int v9; // eax
  unsigned int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int64 *v14; // r15
  unsigned __int64 *v15; // rbx
  __int64 v17; // rax
  int v18; // esi
  unsigned __int64 *v19; // r15
  __m128i *v20; // rdx
  __m128i *v21; // rax
  __int64 v22; // rdi
  unsigned __int64 *v23; // [rsp+8h] [rbp-108h]
  __int64 v24; // [rsp+20h] [rbp-F0h] BYREF
  unsigned __int64 v25; // [rsp+28h] [rbp-E8h]
  __int64 v26; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v27; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 v28; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v29[4]; // [rsp+50h] [rbp-C0h] BYREF
  unsigned __int64 *v30; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v31; // [rsp+78h] [rbp-98h]
  _BYTE v32[32]; // [rsp+80h] [rbp-90h] BYREF
  __int64 v33[2]; // [rsp+A0h] [rbp-70h] BYREF
  _QWORD v34[2]; // [rsp+B0h] [rbp-60h] BYREF
  char v35; // [rsp+C0h] [rbp-50h]
  _QWORD v36[2]; // [rsp+C8h] [rbp-48h] BYREF
  _QWORD *v37; // [rsp+D8h] [rbp-38h] BYREF

  v30 = (unsigned __int64 *)v32;
  v24 = a2;
  v25 = a3;
  v31 = 0x100000000LL;
  if ( a3 )
  {
    do
    {
      v27 = 0;
      v28 = 0;
      LOBYTE(v33[0]) = 59;
      v4 = sub_C931B0(&v24, v33, 1u, 0);
      if ( v4 == -1 )
      {
        v6 = v24;
        v4 = v25;
        v7 = 0;
        v8 = 0;
      }
      else
      {
        v5 = v4 + 1;
        v6 = v24;
        if ( v4 + 1 > v25 )
        {
          v5 = v25;
          v7 = 0;
        }
        else
        {
          v7 = v25 - v5;
        }
        v8 = v24 + v5;
        if ( v4 > v25 )
          v4 = v25;
      }
      v27 = v6;
      v28 = v4;
      v24 = v8;
      v25 = v7;
      if ( v4 <= 0xB || *(_QWORD *)v6 != 0x6576726573657270LL || *(_DWORD *)(v6 + 8) != 1031169837 )
      {
        v9 = sub_C63BB0();
        v33[1] = 41;
        v10 = v9;
        v12 = v11;
        v33[0] = (__int64)"invalid Internalize pass parameter '{0}' ";
        v34[0] = &v37;
        v34[1] = 1;
        v35 = 1;
        v36[0] = &unk_49DB108;
        v36[1] = &v27;
        v37 = v36;
        sub_23328D0((__int64)v29, (__int64)v33);
        sub_23058C0(&v26, (__int64)v29, v10, v12);
        v13 = v26;
        *(_BYTE *)(a1 + 16) |= 3u;
        *(_QWORD *)a1 = v13 & 0xFFFFFFFFFFFFFFFELL;
        sub_2240A30(v29);
        goto LABEL_10;
      }
      v28 = v4 - 12;
      v27 = v6 + 12;
      v33[0] = (__int64)v34;
      sub_2305260(v33, (_BYTE *)(v6 + 12), v6 + v4);
      v17 = (unsigned int)v31;
      v18 = v31;
      if ( (unsigned __int64)(unsigned int)v31 + 1 > HIDWORD(v31) )
      {
        if ( v30 > (unsigned __int64 *)v33 || (v23 = v30, v33 >= (__int64 *)&v30[4 * (unsigned int)v31]) )
        {
          sub_95D880((__int64)&v30, (unsigned int)v31 + 1LL);
          v17 = (unsigned int)v31;
          v19 = v30;
          v20 = (__m128i *)v33;
          v18 = v31;
        }
        else
        {
          sub_95D880((__int64)&v30, (unsigned int)v31 + 1LL);
          v19 = v30;
          v17 = (unsigned int)v31;
          v20 = (__m128i *)((char *)v30 + (char *)v33 - (char *)v23);
          v18 = v31;
        }
      }
      else
      {
        v19 = v30;
        v20 = (__m128i *)v33;
      }
      v21 = (__m128i *)&v19[4 * v17];
      if ( v21 )
      {
        v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
        if ( (__m128i *)v20->m128i_i64[0] == &v20[1] )
        {
          v21[1] = _mm_loadu_si128(v20 + 1);
        }
        else
        {
          v21->m128i_i64[0] = v20->m128i_i64[0];
          v21[1].m128i_i64[0] = v20[1].m128i_i64[0];
        }
        v22 = v20->m128i_i64[1];
        v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
        v20->m128i_i64[1] = 0;
        v21->m128i_i64[1] = v22;
        v18 = v31;
        v20[1].m128i_i8[0] = 0;
      }
      LODWORD(v31) = v18 + 1;
      if ( (_QWORD *)v33[0] != v34 )
        j_j___libc_free_0(v33[0]);
    }
    while ( v25 );
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = *(_BYTE *)(a1 + 16) & 0xFC | 2;
    if ( !(_DWORD)v31 )
      goto LABEL_14;
    sub_23507F0(a1, (__int64)&v30);
LABEL_10:
    v14 = v30;
    v15 = &v30[4 * (unsigned int)v31];
    if ( v30 == v15 )
      goto LABEL_15;
    do
    {
      v15 -= 4;
      if ( (unsigned __int64 *)*v15 != v15 + 2 )
        j_j___libc_free_0(*v15);
    }
    while ( v14 != v15 );
LABEL_14:
    v14 = v30;
LABEL_15:
    if ( v14 != (unsigned __int64 *)v32 )
      _libc_free((unsigned __int64)v14);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = *(_BYTE *)(a1 + 16) & 0xFC | 2;
  }
  return a1;
}

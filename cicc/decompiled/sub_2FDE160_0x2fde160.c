// Function: sub_2FDE160
// Address: 0x2fde160
//
_QWORD *__fastcall sub_2FDE160(_QWORD *a1, __int64 a2, char *a3, __int64 a4, int a5, _QWORD *a6)
{
  unsigned __int64 v9; // rax
  unsigned int v10; // ecx
  char *v11; // rdi
  unsigned __int64 v12; // r14
  int v13; // r8d
  int *v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // r12
  __int64 v18; // rsi
  __int64 v19; // r13
  unsigned __int64 v20; // rcx
  char *v21; // r13
  unsigned int v22; // r15d
  unsigned int v23; // r9d
  const __m128i *v24; // r8
  unsigned int v25; // eax
  char *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // [rsp+8h] [rbp-C8h]
  __int64 v29; // [rsp+10h] [rbp-C0h]
  const __m128i *v30; // [rsp+18h] [rbp-B8h]
  unsigned int v31; // [rsp+18h] [rbp-B8h]
  int v33; // [rsp+24h] [rbp-ACh]
  __int64 v35; // [rsp+38h] [rbp-98h]
  __int64 v36; // [rsp+38h] [rbp-98h]
  unsigned __int64 v39; // [rsp+50h] [rbp-80h]
  __int64 v40; // [rsp+50h] [rbp-80h]
  unsigned int v41; // [rsp+58h] [rbp-78h]
  unsigned int v42; // [rsp+68h] [rbp-68h] BYREF
  unsigned int v43; // [rsp+6Ch] [rbp-64h] BYREF
  __m128i v44; // [rsp+70h] [rbp-60h] BYREF
  __int64 v45; // [rsp+80h] [rbp-50h]
  __int64 v46; // [rsp+88h] [rbp-48h]

  v9 = (*(__int64 (__fastcall **)(_QWORD *))(*a6 + 592LL))(a6);
  v10 = v9;
  v11 = &a3[4 * a4];
  v12 = HIDWORD(v9);
  v35 = 4 * a4;
  v39 = HIDWORD(v9);
  v13 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  v33 = v13;
  if ( a3 == v11 )
  {
LABEL_11:
    v18 = *(_QWORD *)(a2 + 56);
    v44.m128i_i64[0] = v18;
    if ( v18 )
      sub_B96E90((__int64)&v44, v18, 1);
    v16 = sub_2E7B380(a1, a6[1] - 40LL * *(unsigned __int16 *)(a2 + 68), (unsigned __int8 **)&v44, 1u);
    if ( v44.m128i_i64[0] )
      sub_B91220((__int64)&v44, v44.m128i_i64[0]);
    if ( (_DWORD)v39 )
    {
      v19 = 0;
      do
      {
        if ( v33 != (_DWORD)v19 )
          sub_2E8EAD0((__int64)v16, (__int64)a1, (const __m128i *)(*(_QWORD *)(a2 + 32) + 40 * v19));
        ++v19;
      }
      while ( (unsigned int)v39 != v19 );
    }
    v41 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
    if ( (unsigned int)v39 >= v41 )
      return v16;
    v28 = v35 >> 2;
    v20 = v35 & 0xFFFFFFFFFFFFFFF0LL;
    v36 = v35 >> 4;
    v40 = 40 * v39;
    v29 = (v11 - &a3[v20]) >> 2;
    v21 = &a3[v20];
    v22 = v12;
    while ( 1 )
    {
      v23 = v41;
      v24 = (const __m128i *)(*(_QWORD *)(a2 + 32) + v40);
      if ( !v24->m128i_i8[0] && (v24->m128i_i8[3] & 0x10) == 0 && (v24->m128i_i16[1] & 0xFF0) != 0 )
      {
        v30 = (const __m128i *)(*(_QWORD *)(a2 + 32) + v40);
        v25 = sub_2E89F40(a2, v22);
        v24 = v30;
        v23 = v25;
      }
      if ( v36 > 0 )
      {
        v26 = a3;
        while ( v22 != *(_DWORD *)v26 )
        {
          if ( v22 == *((_DWORD *)v26 + 1) )
          {
            v26 += 4;
            goto LABEL_33;
          }
          if ( v22 == *((_DWORD *)v26 + 2) )
          {
            v26 += 8;
            goto LABEL_33;
          }
          if ( v22 == *((_DWORD *)v26 + 3) )
          {
            v26 += 12;
            goto LABEL_33;
          }
          v26 += 16;
          if ( v21 == v26 )
          {
            v27 = v29;
            goto LABEL_39;
          }
        }
        goto LABEL_33;
      }
      v27 = v28;
      v26 = a3;
LABEL_39:
      if ( v27 == 2 )
        goto LABEL_50;
      if ( v27 == 3 )
        break;
      if ( v27 != 1 )
        goto LABEL_43;
LABEL_42:
      if ( v22 != *(_DWORD *)v26 )
      {
LABEL_43:
        v31 = v23;
        sub_2E8EAD0((__int64)v16, (__int64)a1, v24);
        if ( v41 > v31 )
          sub_2E89ED0((__int64)v16, (__PAIR64__(v31, v33) - v31) >> 32, (v16[5] & 0xFFFFFF) - 1);
        goto LABEL_36;
      }
LABEL_33:
      if ( v11 == v26 )
        goto LABEL_43;
      if ( !(*(unsigned __int8 (__fastcall **)(_QWORD *, unsigned __int64, _QWORD, unsigned int *, unsigned int *, _QWORD *))(*a6 + 160LL))(
              a6,
              *(_QWORD *)(*(_QWORD *)(a1[4] + 56LL) + 16LL * (v24->m128i_i32[2] & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
              ((unsigned __int32)v24->m128i_i32[0] >> 8) & 0xFFF,
              &v42,
              &v43,
              a1) )
        sub_C64ED0("cannot spill patchpoint subregister operand", 1u);
      v44.m128i_i64[0] = 1;
      v45 = 0;
      v46 = 1;
      sub_2E8EAD0((__int64)v16, (__int64)a1, &v44);
      v44.m128i_i64[0] = 1;
      v46 = v42;
      v45 = 0;
      sub_2E8EAD0((__int64)v16, (__int64)a1, &v44);
      v44.m128i_i64[0] = 5;
      v45 = 0;
      LODWORD(v46) = a5;
      sub_2E8EAD0((__int64)v16, (__int64)a1, &v44);
      v44.m128i_i64[0] = 1;
      v45 = 0;
      v46 = v43;
      sub_2E8EAD0((__int64)v16, (__int64)a1, &v44);
LABEL_36:
      v40 += 40;
      if ( ++v22 == v41 )
        return v16;
    }
    if ( v22 == *(_DWORD *)v26 )
      goto LABEL_33;
    v26 += 4;
LABEL_50:
    if ( v22 == *(_DWORD *)v26 )
      goto LABEL_33;
    v26 += 4;
    goto LABEL_42;
  }
  v14 = (int *)a3;
  while ( 1 )
  {
    v15 = (unsigned int)*v14;
    if ( (unsigned int)v15 >= v10 )
    {
      if ( (unsigned int)v15 < (unsigned int)v12 )
        return 0;
    }
    else
    {
      v13 = *v14;
    }
    if ( (*(_WORD *)(*(_QWORD *)(a2 + 32) + 40 * v15 + 2) & 0xFF0) != 0 )
      return 0;
    if ( v11 == (char *)++v14 )
    {
      v33 = v13;
      goto LABEL_11;
    }
  }
}

// Function: sub_370D750
// Address: 0x370d750
//
__int64 *__fastcall sub_370D750(__int64 *a1, _QWORD *a2, int **a3, int **a4, char a5)
{
  __m128i *v7; // r14
  unsigned __int64 v8; // rax
  char v9; // r8
  unsigned int v11; // eax
  char v12; // r8
  int **v13; // r10
  unsigned __int64 v14; // rcx
  int *v15; // rax
  int *v16; // r14
  size_t v17; // rbx
  __int64 v18; // rcx
  int *v19; // rsi
  unsigned __int64 v20; // rbx
  __int64 v21; // rcx
  __int64 v22; // r9
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rax
  char v28; // r8
  unsigned __int64 v29; // rax
  int **v30; // [rsp+28h] [rbp-1D8h]
  unsigned __int64 v32; // [rsp+30h] [rbp-1D0h]
  size_t v34; // [rsp+38h] [rbp-1C8h]
  unsigned __int64 v35; // [rsp+48h] [rbp-1B8h] BYREF
  _OWORD v36[2]; // [rsp+50h] [rbp-1B0h] BYREF
  _OWORD v37[2]; // [rsp+70h] [rbp-190h] BYREF
  __m128i v38[2]; // [rsp+90h] [rbp-170h] BYREF
  __int16 v39; // [rsp+B0h] [rbp-150h]
  __m128i v40[2]; // [rsp+C0h] [rbp-140h] BYREF
  __int16 v41; // [rsp+E0h] [rbp-120h]
  __m128i v42; // [rsp+F0h] [rbp-110h] BYREF
  __int64 v43; // [rsp+100h] [rbp-100h]
  _BYTE v44[40]; // [rsp+108h] [rbp-F8h] BYREF
  __m128i v45[2]; // [rsp+130h] [rbp-D0h] BYREF
  __int16 v46; // [rsp+150h] [rbp-B0h]

  if ( !a2[6] || a2[7] || a2[5] )
  {
    v7 = &v42;
    v45[0].m128i_i64[0] = (__int64)"Name";
    v46 = 259;
    sub_3701560((unsigned __int64 *)&v42, a2, a3, v45, a5);
    v8 = v42.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v42.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v42.m128i_i64[0] = 0;
      *a1 = v8 | 1;
      sub_9C66B0(v42.m128i_i64);
      return a1;
    }
    v42.m128i_i64[0] = 0;
    sub_9C66B0(v42.m128i_i64);
    if ( !a5 )
      goto LABEL_5;
    v45[0].m128i_i64[0] = (__int64)"LinkageName";
    v46 = 259;
    goto LABEL_20;
  }
  v11 = sub_3700ED0((__int64)a2, (__int64)a2, (__int64)a3, (__int64)a4, a5);
  v13 = a3;
  v14 = v11;
  if ( !a5 )
  {
    v26 = v11 - 1LL;
    v7 = v40;
    if ( v26 > (unsigned __int64)a3[1] )
      v26 = (unsigned __int64)a3[1];
    v27 = (__int64)*a3;
    v42.m128i_i64[1] = v26;
    v46 = 257;
    v42.m128i_i64[0] = v27;
    sub_3701560((unsigned __int64 *)v40, a2, &v42, v45, v12);
    if ( (v40[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      v40[0].m128i_i64[0] = 0;
      sub_9C66B0(v40[0].m128i_i64);
      goto LABEL_5;
    }
    v40[0].m128i_i64[0] = v40[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL | 1;
LABEL_23:
    *a1 = 0;
    sub_9C6670(a1, v7);
    sub_9C66B0(v7->m128i_i64);
    return a1;
  }
  v15 = a3[1];
  v32 = v14;
  if ( (int *)v14 >= (int *)((char *)v15 + (_QWORD)a4[1] + 2) )
  {
    v7 = &v42;
    v46 = 257;
    sub_3701560((unsigned __int64 *)&v42, a2, v13, v45, v12);
    v25 = v42.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v42.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
LABEL_22:
      v42.m128i_i64[0] = v25 | 1;
      goto LABEL_23;
    }
    v42.m128i_i64[0] = 0;
    sub_9C66B0(v42.m128i_i64);
    v46 = 257;
LABEL_20:
    sub_3701560((unsigned __int64 *)&v42, a2, a4, v45, v9);
    v25 = v42.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v42.m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    {
      v42.m128i_i64[0] = 0;
      sub_9C66B0(v42.m128i_i64);
LABEL_5:
      *a1 = 1;
      v45[0].m128i_i64[0] = 0;
      sub_9C66B0(v45[0].m128i_i64);
      return a1;
    }
    goto LABEL_22;
  }
  v16 = *a4;
  v17 = (size_t)a4[1];
  v30 = v13;
  v42.m128i_i64[0] = (__int64)v44;
  v42.m128i_i64[1] = 0;
  v43 = 32;
  sub_C7D030(v45);
  sub_C7D280(v45[0].m128i_i32, v16, v17);
  sub_C7D290(v45, v40);
  sub_C7D4E0((unsigned __int8 *)v40, &v42);
  v40[0].m128i_i64[0] = (__int64)"@";
  v38[0].m128i_i64[0] = (__int64)"??@";
  v38[1] = v42;
  v39 = 1283;
  v41 = 259;
  sub_9C6370(v45, v38, v40, v18, (__int64)v38, 1283);
  sub_CA0F50((__int64 *)v36, (void **)v45);
  v19 = *v30;
  v20 = 4064;
  v34 = (size_t)v30[1];
  if ( v32 - *((_QWORD *)&v36[0] + 1) - 2 < 0x1000 )
    v20 = v32 - *((_QWORD *)&v36[0] + 1) - 34;
  sub_C7D030(v45);
  sub_C7D280(v45[0].m128i_i32, v19, v34);
  sub_C7D290(v45, v40);
  sub_C7D4E0((unsigned __int8 *)v40, &v42);
  v41 = 261;
  v40[0] = v42;
  v23 = (unsigned __int64)v30[1];
  if ( v20 < v23 )
    v23 = v20;
  v38[0].m128i_i64[0] = (__int64)*v30;
  v39 = 261;
  v38[0].m128i_i64[1] = v23;
  sub_9C6370(v45, v38, v40, v21, (__int64)v38, v22);
  sub_CA0F50((__int64 *)v37, (void **)v45);
  v38[0] = (__m128i)v37[0];
  v46 = 257;
  v40[0] = (__m128i)v36[0];
  sub_3701560(&v35, a2, v38, v45, (unsigned __int8)v38);
  v24 = v35 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v35 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = 0;
    v35 = v24 | 1;
    goto LABEL_17;
  }
  v35 = 0;
  sub_9C66B0((__int64 *)&v35);
  v46 = 257;
  sub_3701560(&v35, a2, v40, v45, v28);
  v29 = v35 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v35 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    v35 = 0;
    sub_9C66B0((__int64 *)&v35);
    sub_2240A30((unsigned __int64 *)v37);
    sub_2240A30((unsigned __int64 *)v36);
    if ( (_BYTE *)v42.m128i_i64[0] != v44 )
      _libc_free(v42.m128i_u64[0]);
    goto LABEL_5;
  }
  *a1 = 0;
  v35 = v29 | 1;
LABEL_17:
  sub_9C6670(a1, &v35);
  sub_9C66B0((__int64 *)&v35);
  sub_2240A30((unsigned __int64 *)v37);
  sub_2240A30((unsigned __int64 *)v36);
  if ( (_BYTE *)v42.m128i_i64[0] != v44 )
    _libc_free(v42.m128i_u64[0]);
  return a1;
}

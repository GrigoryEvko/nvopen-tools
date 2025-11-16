// Function: sub_38A9F20
// Address: 0x38a9f20
//
__int64 __fastcall sub_38A9F20(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // r13
  unsigned int v11; // r12d
  int v13; // eax
  char v14; // r15
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 v17; // rax
  _QWORD *v18; // r8
  __int64 v19; // r15
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // r9
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rsi
  int *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  double v32; // xmm4_8
  double v33; // xmm5_8
  __int64 v34; // rdi
  unsigned __int64 v35; // r13
  const char *v36; // rax
  unsigned __int64 v37; // rsi
  __int64 v38; // r13
  __int64 v39; // rsi
  unsigned __int8 *v40; // r14
  unsigned int v41; // [rsp+14h] [rbp-5Ch] BYREF
  unsigned __int8 *v42; // [rsp+18h] [rbp-58h] BYREF
  unsigned int *v43[2]; // [rsp+20h] [rbp-50h] BYREF
  char v44; // [rsp+30h] [rbp-40h]
  char v45; // [rsp+31h] [rbp-3Fh]

  v9 = a1 + 8;
  v41 = 0;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388BA90(a1, &v41) )
    return 1;
  v11 = sub_388AF10(a1, 3, "expected '=' here");
  if ( (_BYTE)v11 )
    return 1;
  v13 = *(_DWORD *)(a1 + 64);
  if ( v13 == 388 )
  {
    v45 = 1;
    v36 = "unexpected type in metadata definition";
    goto LABEL_32;
  }
  v14 = 0;
  if ( v13 == 300 )
  {
    v14 = 1;
    v13 = sub_3887100(v9);
    *(_DWORD *)(a1 + 64) = v13;
  }
  if ( v13 != 376 )
  {
    if ( !(unsigned __int8)sub_388AF10(a1, 14, "Expected '!' here")
      && !(unsigned __int8)sub_38A2390((__int64 **)a1, (__int64 *)&v42, v14, *(double *)a2.m128_u64, a3, a4) )
    {
      goto LABEL_11;
    }
    return 1;
  }
  if ( (unsigned __int8)sub_38A9970(a1, (__int64 *)&v42, v14, *(double *)a2.m128_u64, a3, a4) )
    return 1;
LABEL_11:
  v17 = *(_QWORD *)(a1 + 872);
  v18 = (_QWORD *)(a1 + 864);
  v19 = a1 + 864;
  if ( !v17 )
    goto LABEL_18;
  do
  {
    while ( 1 )
    {
      v20 = *(_QWORD *)(v17 + 16);
      v21 = *(_QWORD *)(v17 + 24);
      if ( *(_DWORD *)(v17 + 32) >= v41 )
        break;
      v17 = *(_QWORD *)(v17 + 24);
      if ( !v21 )
        goto LABEL_16;
    }
    v19 = v17;
    v17 = *(_QWORD *)(v17 + 16);
  }
  while ( v20 );
LABEL_16:
  if ( (_QWORD *)v19 == v18 || v41 < *(_DWORD *)(v19 + 32) )
  {
LABEL_18:
    v22 = *(_QWORD *)(a1 + 824);
    v23 = a1 + 816;
    if ( !v22 )
    {
      v38 = a1 + 816;
      goto LABEL_34;
    }
    v24 = a1 + 816;
    v25 = *(_QWORD *)(a1 + 824);
    do
    {
      if ( *(_DWORD *)(v25 + 32) < v41 )
      {
        v25 = *(_QWORD *)(v25 + 24);
      }
      else
      {
        v24 = v25;
        v25 = *(_QWORD *)(v25 + 16);
      }
    }
    while ( v25 );
    if ( v23 == v24 || v41 < *(_DWORD *)(v24 + 32) )
    {
      v38 = a1 + 816;
      do
      {
        if ( *(_DWORD *)(v22 + 32) < v41 )
        {
          v22 = *(_QWORD *)(v22 + 24);
        }
        else
        {
          v38 = v22;
          v22 = *(_QWORD *)(v22 + 16);
        }
      }
      while ( v22 );
      if ( v23 != v38 && v41 >= *(_DWORD *)(v38 + 32) )
      {
LABEL_35:
        v39 = *(_QWORD *)(v38 + 40);
        v40 = v42;
        if ( v39 )
          sub_161E7C0(v38 + 40, v39);
        *(_QWORD *)(v38 + 40) = v40;
        if ( v40 )
          sub_1623A60(v38 + 40, (__int64)v40, 2);
        return v11;
      }
LABEL_34:
      v43[0] = &v41;
      v38 = sub_38979A0((_QWORD *)(a1 + 808), v38, v43);
      goto LABEL_35;
    }
    v45 = 1;
    v36 = "Metadata id is already used";
LABEL_32:
    v37 = *(_QWORD *)(a1 + 56);
    v43[0] = (unsigned int *)v36;
    v44 = 3;
    return (unsigned int)sub_38814C0(v9, v37, (__int64)v43);
  }
  v26 = *(_QWORD *)(*(_QWORD *)(v19 + 40) + 16LL);
  if ( (v26 & 4) != 0 )
  {
    sub_16302D0((const __m128i *)(v26 & 0xFFFFFFFFFFFFFFF8LL), v42, a2, a3, a4, a5, v15, v16, a8, a9);
    v18 = (_QWORD *)(a1 + 864);
  }
  v27 = (__int64)v18;
  v28 = sub_220F330((int *)v19, v18);
  v34 = *((_QWORD *)v28 + 5);
  v35 = (unsigned __int64)v28;
  if ( v34 )
    sub_16307F0(v34, v27, v29, v30, v31, a2, a3, a4, a5, v32, v33, a8, a9);
  j_j___libc_free_0(v35);
  --*(_QWORD *)(a1 + 896);
  return v11;
}

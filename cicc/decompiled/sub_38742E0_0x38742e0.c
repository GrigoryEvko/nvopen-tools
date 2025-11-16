// Function: sub_38742E0
// Address: 0x38742e0
//
__int64 __fastcall sub_38742E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  __int64 v17; // r15
  _QWORD *v18; // rax
  _QWORD *v19; // r9
  int v20; // eax
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // r9
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rsi
  unsigned __int64 v29; // rdi
  __int64 v30; // rsi
  const char *v31; // rax
  __int64 v32; // rdx
  __int64 v34; // [rsp+10h] [rbp-80h]
  __int64 v35; // [rsp+20h] [rbp-70h]
  __int64 v36; // [rsp+20h] [rbp-70h]
  __int64 v37; // [rsp+20h] [rbp-70h]
  _QWORD v39[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v40[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v41; // [rsp+50h] [rbp-40h]

  v17 = *(_QWORD *)(a2 + 8);
  if ( !v17 )
  {
LABEL_19:
    if ( a5 )
      a5 -= 24;
    v31 = sub_1649960(a2);
    v39[1] = v32;
    v39[0] = v31;
    v41 = 261;
    v40[0] = v39;
    v27 = sub_15FDBD0(a4, a2, a3, (__int64)v40, a5);
    goto LABEL_22;
  }
  while ( 1 )
  {
    v18 = sub_1648700(v17);
    v19 = v18;
    if ( a3 == *v18 )
    {
      v20 = *((unsigned __int8 *)v18 + 16);
      if ( (unsigned __int8)v20 > 0x17u && (unsigned int)(v20 - 60) <= 0xC && a4 == v20 - 24 )
        break;
    }
    v17 = *(_QWORD *)(v17 + 8);
    if ( !v17 )
      goto LABEL_19;
  }
  if ( (_QWORD *)a5 == v19 + 3 )
  {
    v27 = (__int64)v19;
    if ( a5 != *(_QWORD *)(a1 + 280) )
      goto LABEL_22;
  }
  else if ( !a5 )
  {
    v21 = 0;
    goto LABEL_11;
  }
  v21 = a5 - 24;
LABEL_11:
  v35 = (__int64)v19;
  v41 = 257;
  v22 = sub_15FDBD0(a4, a2, a3, (__int64)v40, v21);
  v23 = v35;
  v36 = v22;
  v34 = v23;
  sub_164B7C0(v22, v23);
  sub_164D160(v34, v36, a6, a7, a8, a9, v24, v25, a12, a13);
  v26 = sub_1599EF0(*(__int64 ***)a2);
  v27 = v36;
  if ( *(_QWORD *)(v34 - 24) )
  {
    v28 = *(_QWORD *)(v34 - 16);
    v29 = *(_QWORD *)(v34 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v29 = v28;
    if ( v28 )
      *(_QWORD *)(v28 + 16) = v29 | *(_QWORD *)(v28 + 16) & 3LL;
  }
  *(_QWORD *)(v34 - 24) = v26;
  if ( v26 )
  {
    v30 = *(_QWORD *)(v26 + 8);
    *(_QWORD *)(v34 - 16) = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 16) = (v34 - 16) | *(_QWORD *)(v30 + 16) & 3LL;
    *(_QWORD *)(v34 - 24 + 16) = (v26 + 8) | *(_QWORD *)(v34 - 8) & 3LL;
    *(_QWORD *)(v26 + 8) = v34 - 24;
  }
  if ( !v36 )
    goto LABEL_19;
LABEL_22:
  v37 = v27;
  sub_38740E0(a1, v27);
  return v37;
}

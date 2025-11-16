// Function: sub_3878BC0
// Address: 0x3878bc0
//
__int64 __fastcall sub_3878BC0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11,
        __int64 a12,
        __int64 a13,
        __int64 **a14,
        char a15)
{
  char *v17; // rdx
  char v18; // cl
  __int64 *v19; // rdi
  __int64 v20; // r12
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v25; // rax
  double v26; // xmm4_8
  double v27; // xmm5_8
  _QWORD *v28; // rax
  __int64 **v29; // rcx
  int v30; // r15d
  _QWORD *v31; // rax
  __int64 *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 *v35; // rbx
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38[2]; // [rsp+0h] [rbp-70h] BYREF
  __int16 v39; // [rsp+10h] [rbp-60h]
  __int64 v40[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v41; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a13 + 8) == 15 )
  {
    v23 = a13;
    if ( *(_BYTE *)(a3 + 16) != 13 )
    {
      v30 = *(_DWORD *)(a13 + 8) >> 8;
      v31 = (_QWORD *)sub_15E0530(*(_QWORD *)(*a1 + 24));
      v32 = (__int64 *)sub_1643320(v31);
      v23 = sub_1646BA0(v32, v30);
    }
    v25 = sub_146F1B0(*a1, a3);
    v28 = (_QWORD *)sub_3878B90(a1, v25, v23, a14, (__int64 ***)a2, a4, a5, a6, a7, v26, v27, a10, a11);
    v29 = *(__int64 ***)a2;
    v20 = (__int64)v28;
    if ( *(_QWORD *)a2 != *v28 )
    {
      v41 = 257;
      v20 = (__int64)sub_38723F0(a1 + 33, 47, (__int64)v28, v29, v40);
      sub_38740E0((__int64)a1, v20);
    }
  }
  else
  {
    v17 = (char *)a1[2];
    v18 = *v17;
    if ( a15 )
    {
      v19 = a1 + 33;
      if ( v18 )
      {
        v40[0] = (__int64)v17;
        v40[1] = (__int64)".iv.next";
        v41 = 771;
      }
      else
      {
        v40[0] = (__int64)".iv.next";
        v41 = 259;
      }
      v20 = (__int64)sub_38718D0(v19, a2, a3, v40, 0, 0, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
    }
    else
    {
      if ( v18 )
      {
        v38[0] = a1[2];
        v38[1] = (__int64)".iv.next";
        v39 = 771;
      }
      else
      {
        v38[0] = (__int64)".iv.next";
        v39 = 259;
      }
      if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
      {
        v41 = 257;
        v33 = sub_15FB440(11, (__int64 *)a2, a3, (__int64)v40, 0);
        v34 = a1[34];
        v20 = v33;
        if ( v34 )
        {
          v35 = (__int64 *)a1[35];
          sub_157E9D0(v34 + 40, v33);
          v36 = *(_QWORD *)(v20 + 24);
          v37 = *v35;
          *(_QWORD *)(v20 + 32) = v35;
          v37 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v20 + 24) = v37 | v36 & 7;
          *(_QWORD *)(v37 + 8) = v20 + 24;
          *v35 = *v35 & 7 | (v20 + 24);
        }
        sub_164B780(v20, v38);
        sub_12A86E0(a1 + 33, v20);
      }
      else
      {
        v20 = sub_15A2B30((__int64 *)a2, a3, 0, 0, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v22 = sub_14DBA30(v20, a1[41], 0);
        if ( v22 )
          v20 = v22;
      }
    }
    sub_38740E0((__int64)a1, v20);
  }
  return v20;
}

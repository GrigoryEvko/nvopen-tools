// Function: sub_1BADBC0
// Address: 0x1badbc0
//
unsigned __int64 __fastcall sub_1BADBC0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // r14
  _QWORD *v13; // r12
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  int v16; // esi
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned __int64 v20; // rax
  __int64 v21; // r9
  _QWORD *v22; // rax
  _QWORD *v23; // r14
  unsigned __int64 v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  int v27; // r8d
  int v28; // r9d
  unsigned __int64 result; // rax
  __int64 v30; // rsi
  _QWORD *v31; // rax
  _QWORD **v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 v35; // r9
  __int64 *v36; // r14
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // [rsp+0h] [rbp-F0h]
  __int64 v40; // [rsp+8h] [rbp-E8h]
  __int16 v41; // [rsp+14h] [rbp-DCh]
  __int64 v43; // [rsp+20h] [rbp-D0h]
  __int64 v44; // [rsp+20h] [rbp-D0h]
  __int64 v45; // [rsp+20h] [rbp-D0h]
  _QWORD *v46; // [rsp+20h] [rbp-D0h]
  const char *v47; // [rsp+30h] [rbp-C0h] BYREF
  char v48; // [rsp+40h] [rbp-B0h]
  char v49; // [rsp+41h] [rbp-AFh]
  _QWORD v50[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v51; // [rsp+60h] [rbp-90h]
  __int64 v52; // [rsp+70h] [rbp-80h] BYREF
  __int64 v53; // [rsp+78h] [rbp-78h]
  __int64 *v54; // [rsp+80h] [rbp-70h]
  __int64 v55; // [rsp+88h] [rbp-68h]
  __int64 v56; // [rsp+90h] [rbp-60h]
  int v57; // [rsp+98h] [rbp-58h]
  __int64 v58; // [rsp+A0h] [rbp-50h]
  __int64 v59; // [rsp+A8h] [rbp-48h]

  v12 = sub_1B91F20((_QWORD *)a1, (__int64)a2, a4, a5);
  v13 = (_QWORD *)sub_13FC520((__int64)a2);
  v14 = sub_157EBA0((__int64)v13);
  v52 = 0;
  v55 = sub_16498A0(v14);
  v54 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v53 = 0;
  sub_17050D0(&v52, v14);
  v15 = *(_QWORD *)(a1 + 456);
  v16 = *(_DWORD *)(a1 + 88);
  v49 = 1;
  v48 = 3;
  v17 = (unsigned int)(*(_DWORD *)(a1 + 92) * v16);
  v41 = 36 - ((*(_BYTE *)(*(_QWORD *)(v15 + 384) + 40LL) == 0) - 1);
  v47 = "min.iters.check";
  v18 = sub_15A0680(*(_QWORD *)v12, v17, 0);
  if ( *(_BYTE *)(v12 + 16) > 0x10u || *(_BYTE *)(v18 + 16) > 0x10u )
  {
    v45 = v18;
    v51 = 257;
    v31 = sub_1648A60(56, 2u);
    v19 = (__int64)v31;
    if ( v31 )
    {
      v40 = (__int64)v31;
      v32 = *(_QWORD ***)v12;
      if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16 )
      {
        v39 = v45;
        v46 = v32[4];
        v33 = (__int64 *)sub_1643320(*v32);
        v34 = (__int64)sub_16463B0(v33, (unsigned int)v46);
        v35 = v39;
      }
      else
      {
        v34 = sub_1643320(*v32);
        v35 = v45;
      }
      sub_15FEC10(v19, v34, 51, v41, v12, v35, (__int64)v50, 0);
    }
    else
    {
      v40 = 0;
    }
    if ( v53 )
    {
      v36 = v54;
      sub_157E9D0(v53 + 40, v19);
      v37 = *(_QWORD *)(v19 + 24);
      v38 = *v36;
      *(_QWORD *)(v19 + 32) = v36;
      v38 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v19 + 24) = v38 | v37 & 7;
      *(_QWORD *)(v38 + 8) = v19 + 24;
      *v36 = *v36 & 7 | (v19 + 24);
    }
    sub_164B780(v40, (__int64 *)&v47);
    sub_12A86E0(&v52, v19);
  }
  else
  {
    v19 = sub_15A37B0(v41, (_QWORD *)v12, (_QWORD *)v18, 0);
  }
  v50[0] = "vector.ph";
  v51 = 259;
  v20 = sub_157EBA0((__int64)v13);
  v43 = sub_157FBF0(v13, (__int64 *)(v20 + 24), (__int64)v50);
  sub_1BACEB0(*(_QWORD *)(a1 + 32), v43, (__int64)v13);
  v21 = v43;
  if ( *a2 )
  {
    sub_1400330(*a2, v43, *(_QWORD *)(a1 + 24));
    v21 = v43;
  }
  v44 = v21;
  v22 = sub_1648A60(56, 3u);
  v23 = v22;
  if ( v22 )
    sub_15F83E0((__int64)v22, a3, v44, v19, 0);
  v24 = sub_157EBA0((__int64)v13);
  sub_1AA6530(v24, v23, (__m128)a4, *(double *)a5.m128i_i64, a6, a7, v25, v26, a10, a11);
  result = *(unsigned int *)(a1 + 224);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 228) )
  {
    sub_16CD150(a1 + 216, (const void *)(a1 + 232), 0, 8, v27, v28);
    result = *(unsigned int *)(a1 + 224);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 216) + 8 * result) = v13;
  v30 = v52;
  ++*(_DWORD *)(a1 + 224);
  if ( v30 )
    return sub_161E7C0((__int64)&v52, v30);
  return result;
}

// Function: sub_2AECC70
// Address: 0x2aecc70
//
char *__fastcall sub_2AECC70(__int64 a1, unsigned __int64 a2, unsigned int a3)
{
  __int64 *v4; // r14
  int v5; // ebx
  int v6; // r13d
  _BYTE *v7; // r12
  int v8; // eax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // r14
  char v15; // al
  __m128i v16; // rax
  unsigned int v17; // r13d
  unsigned int v18; // r9d
  __int32 v19; // r12d
  _QWORD *v20; // rdi
  __int64 *v21; // r14
  __int64 v22; // rax
  __int64 v23; // r10
  unsigned int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // rsi
  _QWORD *v27; // rax
  bool v28; // zf
  int v29; // eax
  __int64 v30; // rdx
  __int64 *v31; // rax
  int v32; // eax
  int v33; // eax
  __m128i v34; // xmm1
  int v35; // r12d
  __int8 v36; // cl
  unsigned int v37; // [rsp+Ch] [rbp-A4h]
  __int64 v38; // [rsp+18h] [rbp-98h]
  unsigned int v39; // [rsp+18h] [rbp-98h]
  _QWORD *v40; // [rsp+18h] [rbp-98h]
  __int64 v41; // [rsp+20h] [rbp-90h]
  _QWORD *v42; // [rsp+20h] [rbp-90h]
  unsigned int v43; // [rsp+28h] [rbp-88h]
  __int8 v44; // [rsp+28h] [rbp-88h]
  __m128i v46; // [rsp+50h] [rbp-60h] BYREF
  __int128 v47; // [rsp+60h] [rbp-50h] BYREF
  _QWORD v48[8]; // [rsp+70h] [rbp-40h] BYREF

  if ( **(_BYTE **)(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 56LL) + 8LL)
    && (unsigned __int8)sub_DF9710(*(_QWORD *)(a1 + 448)) )
  {
    sub_2AB8760(
      (__int64)"Not inserting runtime ptr check for divergent target",
      52,
      "runtime pointer checks needed. Not enabled for divergent target",
      0x3Fu,
      (__int64)"CantVersionLoopWithDivergentTarget",
      34,
      *(__int64 **)(a1 + 480),
      *(_QWORD *)(a1 + 416),
      0);
    goto LABEL_15;
  }
  v4 = *(__int64 **)(*(_QWORD *)(a1 + 424) + 112LL);
  v5 = sub_DCF980(v4, *(char **)(a1 + 416));
  v6 = sub_DEF800(*(_QWORD *)(a1 + 424));
  if ( v5 == 1 )
  {
    sub_2AB8760(
      (__int64)"Single iteration (non) loop",
      27,
      "loop trip count is one, irrelevant for vectorization",
      0x34u,
      (__int64)"SingleIterationLoop",
      19,
      *(__int64 **)(a1 + 480),
      *(_QWORD *)(a1 + 416),
      0);
LABEL_15:
    LODWORD(v47) = 0;
    BYTE4(v47) = 0;
    DWORD2(v47) = 0;
    BYTE12(v47) = 1;
    return (char *)v47;
  }
  v7 = (_BYTE *)sub_DCF3A0(v4, *(char **)(a1 + 416), 0);
  if ( !sub_D96A50((__int64)v7) )
  {
    v11 = sub_D95540((__int64)v7);
    v43 = sub_BCB060(v11);
    if ( v43 >= (unsigned int)sub_BCB060(*(_QWORD *)(*(_QWORD *)(a1 + 440) + 336LL)) )
    {
      v12 = sub_D95540((__int64)v7);
      v13 = sub_DA2C50((__int64)v4, v12, -1, 1u);
      if ( (unsigned __int8)sub_DC3A60((__int64)v4, 32, v7, v13) )
      {
        sub_2AB8760(
          (__int64)"Trip count computation wrapped",
          30,
          "backedge-taken count is -1, loop trip count wrapped to 0",
          0x38u,
          (__int64)"TripCountWrapped",
          16,
          *(__int64 **)(a1 + 480),
          *(_QWORD *)(a1 + 416),
          0);
        goto LABEL_15;
      }
    }
  }
  v8 = *(_DWORD *)(a1 + 96);
  if ( v8 )
  {
    if ( (unsigned int)(v8 - 1) <= 1 && (unsigned __int8)sub_2AB8950(a1) )
      goto LABEL_8;
    v10 = sub_D46F00(*(_QWORD *)(a1 + 416));
    if ( v10 != sub_D47930(*(_QWORD *)(a1 + 416)) )
    {
      if ( *(_DWORD *)(a1 + 96) == 3 )
      {
        *(_DWORD *)(a1 + 96) = 0;
        return sub_2AEC460(a1, v6, a2, 0);
      }
LABEL_8:
      LODWORD(v47) = 0;
      BYTE4(v47) = 0;
      DWORD2(v47) = 0;
      BYTE12(v47) = 1;
      return (char *)v47;
    }
    v14 = *(_QWORD *)(a1 + 448);
    if ( (int)sub_23DF0D0(dword_500E228) <= 0 )
      v15 = sub_DFAE30(v14);
    else
      v15 = byte_500E2A8;
    if ( !v15 )
      sub_9BA4A0(*(_QWORD *)(a1 + 504));
    v16.m128i_i64[0] = (__int64)sub_2AEC460(a1, v6, a2, 1);
    v46 = v16;
    v17 = v16.m128i_i32[0];
    v18 = v16.m128i_i32[0];
    v19 = v16.m128i_i32[2];
    v44 = v16.m128i_i8[4];
    if ( v16.m128i_i32[2] )
    {
      *(_QWORD *)&v47 = sub_2AA7E40(*(_QWORD *)(a1 + 488), *(_QWORD *)(a1 + 448));
      if ( !BYTE4(v47) || !(unsigned __int8)sub_DFB2A0(*(_QWORD *)(a1 + 448)) )
        goto LABEL_32;
      v18 = v19 * v47;
      if ( v19 * (int)v47 < v17 )
        v18 = v17;
    }
    if ( v18 )
    {
      if ( a3 )
        v18 *= a3;
      v20 = *(_QWORD **)(a1 + 424);
      v37 = v18;
      v21 = (__int64 *)v20[14];
      v41 = sub_DEF9D0(v20);
      v22 = sub_D95540(v41);
      v48[1] = sub_DA2C50((__int64)v21, v22, 1, 0);
      *(_QWORD *)&v47 = v48;
      v48[0] = v41;
      v38 = v41;
      *((_QWORD *)&v47 + 1) = 0x200000002LL;
      v42 = sub_DC7EB0(v21, (__int64)&v47, 0, 0);
      v23 = v38;
      v24 = v37;
      if ( (_QWORD *)v47 != v48 )
      {
        _libc_free(v47);
        v23 = v38;
        v24 = v37;
      }
      v39 = v24;
      v25 = sub_D95540(v23);
      v40 = sub_DA2C50((__int64)v21, v25, v39, 0);
      v26 = sub_DE4F70(v21, (__int64)v42, *(_QWORD *)(a1 + 416));
      v27 = sub_DCFA50(v21, v26, (__int64)v40);
      if ( sub_D968A0((__int64)v27) )
        return (char *)_mm_load_si128(&v46).m128i_u64[0];
    }
LABEL_32:
    if ( !(unsigned __int8)sub_31A9F60(*(_QWORD *)(a1 + 440)) )
    {
      v28 = *(_BYTE *)(a1 + 108) == 0;
      *(_QWORD *)(a1 + 100) = 0;
      if ( v28 )
        *(_BYTE *)(a1 + 108) = 1;
LABEL_35:
      v29 = *(_DWORD *)(a1 + 96);
      if ( v29 == 3 )
      {
        v34 = _mm_load_si128(&v46);
        *(_DWORD *)(a1 + 96) = 0;
        return (char *)*(_OWORD *)&v34;
      }
      if ( v29 != 4 )
      {
        v30 = *(_QWORD *)(a1 + 416);
        v31 = *(__int64 **)(a1 + 480);
        if ( v5 )
          sub_2AB8760(
            (__int64)"Cannot optimize for size and vectorize at the same time.",
            56,
            "cannot optimize for size and vectorize at the same time. Enable vectorization of this loop with '#pragma cla"
            "ng loop vectorize(enable)' when compiling with -Os/-Oz",
            0xA2u,
            (__int64)"NoTailLoopWithOptForSize",
            24,
            v31,
            v30,
            0);
        else
          sub_2AB8760(
            (__int64)"unable to calculate the loop count due to complex control flow",
            62,
            "unable to calculate the loop count due to complex control flow",
            0x3Eu,
            (__int64)"UnknownLoopCountComplexCFG",
            26,
            v31,
            v30,
            0);
        goto LABEL_15;
      }
      goto LABEL_8;
    }
    if ( (unsigned int)sub_23DF0D0(dword_500E4C8) )
    {
      v32 = dword_500E548;
      v28 = *(_BYTE *)(a1 + 108) == 0;
      *(_DWORD *)(a1 + 100) = dword_500E548;
      *(_DWORD *)(a1 + 104) = v32;
      if ( v28 )
        *(_BYTE *)(a1 + 108) = 1;
      if ( dword_500E548 == 5 )
      {
        if ( !v19 || a3 > 1 || !(unsigned __int8)sub_DFE670(*(_QWORD *)(a1 + 448)) || LOBYTE(qword_500D340[17]) )
        {
          v28 = *(_BYTE *)(a1 + 108) == 0;
          *(_QWORD *)(a1 + 100) = 0x200000002LL;
          if ( v28 )
            *(_BYTE *)(a1 + 108) = 1;
          goto LABEL_49;
        }
        if ( !*(_BYTE *)(a1 + 108) )
          goto LABEL_35;
      }
      v33 = *(_DWORD *)(a1 + 100);
    }
    else
    {
      v35 = sub_DF9CF0(*(__int64 **)(a1 + 448), 0);
      v33 = sub_DF9CF0(*(__int64 **)(a1 + 448), 1u);
      v28 = *(_BYTE *)(a1 + 108) == 0;
      *(_DWORD *)(a1 + 104) = v35;
      *(_DWORD *)(a1 + 100) = v33;
      if ( v28 )
        *(_BYTE *)(a1 + 108) = 1;
    }
    if ( !v33 )
      goto LABEL_35;
    v36 = v44;
    if ( v33 == 5 )
    {
      v36 = 0;
      v17 = 1;
    }
    v44 = v36;
LABEL_49:
    v46.m128i_i32[0] = v17;
    v46.m128i_i8[4] = v44;
    return (char *)_mm_load_si128(&v46).m128i_u64[0];
  }
  return sub_2AEC460(a1, v6, a2, 0);
}

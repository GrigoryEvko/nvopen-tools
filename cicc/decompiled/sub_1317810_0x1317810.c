// Function: sub_1317810
// Address: 0x1317810
//
__int64 __fastcall sub_1317810(_BYTE *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rdi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  unsigned int v12; // r15d
  char *v13; // rcx
  __m128i v15; // xmm0
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-78h]
  __int64 v18; // [rsp+18h] [rbp-68h] BYREF
  __m128i v19; // [rsp+20h] [rbp-60h] BYREF
  __m128i v20; // [rsp+30h] [rbp-50h]
  __int64 v21; // [rsp+40h] [rbp-40h]

  if ( (_DWORD)a2 )
  {
    v4 = sub_131BF20(a1, a2, *(_QWORD *)a3, *(unsigned __int8 *)(a3 + 8));
    if ( !v4 )
      return 0;
    v9 = sub_131C440(a1, v4, 224LL * (unsigned int)dword_4F96B60 + 78984, 64);
    v6 = v9;
    if ( !v9 )
      goto LABEL_9;
    *(_DWORD *)v9 = 0;
    v10 = v9 + 10408;
    *(_DWORD *)(v9 + 4) = 0;
    *(_QWORD *)(v9 + 16) = 0;
    *(_QWORD *)(v9 + 10392) = 0;
    *(_QWORD *)(v9 + 10400) = 0;
    if ( (unsigned __int8)sub_130AF40(v9 + 10408)
      || (*(_DWORD *)(v6 + 10520) = sub_1346400(v10, "tcache_ql"),
          *(_QWORD *)(v6 + 10528) = 0,
          (unsigned __int8)sub_130AF40(v6 + 10536)) )
    {
LABEL_9:
      sub_131C100(a1, v4);
      return 0;
    }
  }
  else
  {
    v4 = sub_131BF10();
    v5 = sub_131C440(a1, v4, 224LL * (unsigned int)dword_4F96B60 + 78984, 64);
    v6 = v5;
    if ( !v5 )
      return 0;
    *(_DWORD *)v5 = 0;
    v7 = v5 + 10408;
    *(_DWORD *)(v5 + 4) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_QWORD *)(v5 + 10392) = 0;
    *(_QWORD *)(v5 + 10400) = 0;
    if ( (unsigned __int8)sub_130AF40(v5 + 10408) )
      return 0;
    *(_DWORD *)(v6 + 10520) = sub_1346400(v7, "tcache_ql");
    *(_QWORD *)(v6 + 10528) = 0;
    if ( (unsigned __int8)sub_130AF40(v6 + 10536) )
      return 0;
  }
  sub_130B270(&v18);
  v17 = sub_1317030(&v18, "arena_large");
  v11 = sub_1316FF0();
  if ( (unsigned __int8)sub_130B2C0(
                          (int)a1,
                          v6 + 10648,
                          (__int64)&unk_5260B60,
                          (__int64)&unk_5060AE0,
                          v4,
                          a2,
                          (_OWORD *)(v6 + 112),
                          0,
                          (__int64)&v18,
                          unk_4C6F220,
                          v11,
                          v17) )
    goto LABEL_18;
  *(_DWORD *)(v6 + 8) = 0;
  if ( dword_4F96B60 )
  {
    v12 = 0;
    while ( !(unsigned __int8)sub_131C760(v6 + 224LL * v12 + 78984) )
    {
      if ( dword_4F96B60 <= ++v12 )
        goto LABEL_20;
    }
LABEL_18:
    if ( !(_DWORD)a2 )
      return 0;
    goto LABEL_9;
  }
LABEL_20:
  *(_QWORD *)(v6 + 78936) = v4;
  sub_1300B60(a2, v6);
  *(_DWORD *)(v6 + 78928) = a2;
  v13 = "auto";
  if ( (unsigned int)a2 >= dword_5057900[0] )
    v13 = "manual";
  sub_40E1DF(v6 + 78952, 0x20u, "%s_%u", v13, (unsigned int)a2);
  *(_BYTE *)(v6 + 78983) = 0;
  sub_130B270((__int64 *)(v6 + 78944));
  if ( byte_4F9698C[0]
    && *(__int64 (__fastcall ***)(int, int, int, int, int, int, int))(sub_131C0E0(v4) + 8) == &off_49E8020 )
  {
    if ( !(_DWORD)a2 )
      return v6;
    v15 = _mm_loadu_si128((const __m128i *)qword_4C6F080);
    v16 = qword_4C6F080[4];
    v20 = _mm_loadu_si128((const __m128i *)&qword_4C6F080[2]);
    v21 = v16;
    v19 = v15;
    v20.m128i_i8[4] = byte_5260DD0[0];
    if ( (unsigned __int8)sub_130B3C0((__int64)a1, v6 + 10648, (__int64)&v19, (__int64)qword_4C6F040) )
      goto LABEL_9;
  }
  else if ( !(_DWORD)a2 )
  {
    return v6;
  }
  ++a1[1];
  if ( !a1[816] )
    sub_1313A40(a1);
  if ( unk_4F96DA8 )
    unk_4F96DA8();
  if ( a1[1]-- == 1 )
    sub_1313A40(a1);
  return v6;
}

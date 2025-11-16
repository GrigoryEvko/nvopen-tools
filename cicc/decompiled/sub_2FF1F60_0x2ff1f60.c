// Function: sub_2FF1F60
// Address: 0x2ff1f60
//
void (*__fastcall sub_2FF1F60(__int64 a1))()
{
  void (*v2)(); // rax
  bool v3; // zf
  __int64 v4; // rax
  void (*v5)(); // rax
  void (*v6)(); // rax
  __int64 (*v7)(); // rax
  __int64 (__fastcall *v8)(__int64); // rax
  void (*v9)(); // rax
  __int64 v10; // rsi
  int v11; // eax
  _QWORD *v12; // rsi
  void (*v13)(); // rax
  __int64 v14; // r12
  void (__fastcall *v15)(__int64, __int64, _QWORD); // r13
  __int64 v16; // rax
  void (*result)(); // rax
  __int64 v18; // rsi
  _QWORD *v19; // rsi
  __int64 v20; // rdi
  _QWORD *v21; // rsi
  _QWORD *v22; // rax
  __int64 (__fastcall *v23)(__int64); // rax
  _QWORD *v24; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // rsi
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rsi
  _QWORD *v31; // rsi
  __int64 v32; // rdi
  _QWORD *v33; // rsi
  _QWORD *v34; // rax
  _QWORD *v35; // rsi
  __int64 v36; // rax
  __m128i *v37; // rdx
  __m128i si128; // xmm0
  __m128i v39; // xmm0
  _QWORD *v40; // rsi
  __int64 v41; // [rsp+8h] [rbp-98h] BYREF
  _BYTE *v42; // [rsp+10h] [rbp-90h] BYREF
  __int64 v43; // [rsp+18h] [rbp-88h]
  _QWORD v44[2]; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 v45[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v46[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v47[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v48[8]; // [rsp+60h] [rbp-40h] BYREF

  *(_BYTE *)(a1 + 250) = 1;
  if ( (unsigned int)sub_2FF0570(a1) )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 288LL))(a1);
  else
    sub_2FF12A0(a1, &unk_501EB24, 0);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 256) + 878LL) & 0x20) != 0 )
  {
    v24 = (_QWORD *)sub_2F7C8B0();
    sub_2FF0E80(a1, v24, 0);
  }
  v2 = *(void (**)())(*(_QWORD *)a1 + 304LL);
  if ( v2 != nullsub_1692 )
    ((void (__fastcall *)(__int64))v2)(a1);
  *(_BYTE *)(a1 + 251) = 0;
  if ( LOBYTE(qword_4F813A8[8]) )
  {
    v29 = (_QWORD *)sub_3581200(1);
    sub_2FF0E80(a1, v29, 0);
    sub_2FEF140((__int64)&v42, *(_QWORD *)(a1 + 256));
    if ( v43 && !(_BYTE)qword_5028008 )
    {
      v30 = *(_QWORD *)(a1 + 256);
      v41 = 0;
      sub_2FEF0A0((__int64)v45, v30);
      v47[0] = (__int64)v48;
      sub_2FEEBD0(v47, v42, (__int64)&v42[v43]);
      v31 = (_QWORD *)sub_3585380(v47, v45, 1, &v41);
      sub_2FF0E80(a1, v31, 0);
      if ( (_QWORD *)v47[0] != v48 )
        j_j___libc_free_0(v47[0]);
      if ( (_QWORD *)v45[0] != v46 )
        j_j___libc_free_0(v45[0]);
      v32 = v41;
      if ( v41 && !_InterlockedSub((volatile signed __int32 *)(v41 + 8), 1u) )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 8LL))(v32);
    }
    if ( v42 != (_BYTE *)v44 )
      j_j___libc_free_0((unsigned __int64)v42);
  }
  v3 = (unsigned __int8)sub_2FF1F10(a1) == 0;
  v4 = *(_QWORD *)a1;
  if ( v3 )
  {
    v23 = *(__int64 (__fastcall **)(__int64))(v4 + 320);
    if ( v23 == sub_2FF1400 )
    {
      sub_2FF12A0(a1, &unk_5022C2C, 0);
      sub_2FF12A0(a1, &unk_502A48C, 0);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 432LL))(a1);
    }
    else
    {
      v23(a1);
    }
  }
  else
  {
    (*(void (__fastcall **)(__int64))(v4 + 328))(a1);
  }
  v5 = *(void (**)())(*(_QWORD *)a1 + 360LL);
  if ( v5 != nullsub_1694 )
    ((void (__fastcall *)(__int64))v5)(a1);
  sub_2FF12A0(a1, &unk_5024F58, 0);
  sub_2FF12A0(a1, &unk_501D684, 0);
  if ( (unsigned int)sub_2FF0570(a1) )
  {
    sub_2FF12A0(a1, &unk_5021D24, 0);
    sub_2FF12A0(a1, &unk_503FF48, 0);
  }
  if ( !(unsigned __int8)sub_2FF0C20(a1, &unk_503FCFC) )
  {
    v27 = (_QWORD *)sub_35AE040();
    sub_2FF0E80(a1, v27, 0);
  }
  if ( (unsigned int)sub_2FF0570(a1) )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 368LL))(a1);
  sub_2FF12A0(a1, &unk_501D66C, 1u);
  v6 = *(void (**)())(*(_QWORD *)a1 + 376LL);
  if ( v6 != nullsub_1695 )
    ((void (__fastcall *)(__int64))v6)(a1);
  if ( (_BYTE)qword_5028D88 )
    sub_2FF12A0(a1, &unk_503BAEC, 0);
  if ( (unsigned int)sub_2FF0570(a1) )
  {
    v7 = *(__int64 (**)())(**(_QWORD **)(a1 + 256) + 152LL);
    if ( v7 == sub_23CE330 || !(unsigned __int8)v7() )
    {
      if ( (_BYTE)qword_5027C48 )
        sub_2FF12A0(a1, &unk_502106C, 0);
      else
        sub_2FF12A0(a1, &unk_5022FAC, 0);
    }
  }
  v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 384LL);
  if ( v8 == sub_2FF1660 )
    sub_2FF12A0(a1, &unk_501DA24, 0);
  else
    v8(a1);
  if ( (unsigned int)sub_2FF0570(a1) )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 392LL))(a1);
  sub_2FF12A0(a1, &unk_503B124, 0);
  sub_2FF12A0(a1, &unk_502A90C, 0);
  sub_2FF12A0(a1, &unk_50226EC, 0);
  v9 = *(void (**)())(*(_QWORD *)a1 + 400LL);
  if ( v9 != nullsub_1696 )
    ((void (__fastcall *)(__int64))v9)(a1);
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 256) + 878LL) & 0x20) != 0 )
  {
    v28 = (_QWORD *)sub_2F7B7D0();
    sub_2FF0E80(a1, v28, 0);
  }
  sub_2FF12A0(a1, &unk_503B12C, 0);
  sub_2FF12A0(a1, &unk_503FF3C, 0);
  sub_2FF12A0(a1, &unk_5040114, 0);
  sub_2FF12A0(a1, &unk_503A0EC, 0);
  sub_2FF12A0(a1, &unk_503FF3E, 0);
  if ( *(char *)(*(_QWORD *)(a1 + 256) + 878LL) < 0
    && (unsigned int)sub_2FF0570(a1)
    && dword_50287A8 != 2
    && (dword_50287A8 == 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 256) + 879LL) & 4) != 0) )
  {
    v33 = (_QWORD *)sub_35348F0(dword_50287A8 == 1);
    sub_2FF0E80(a1, v33, 0);
  }
  if ( (_BYTE)qword_5027308 )
  {
    v26 = (_QWORD *)sub_34E6090();
    sub_2FF0E80(a1, v26, 0);
  }
  if ( LOBYTE(qword_4F813A8[8]) )
  {
    v25 = (_QWORD *)sub_3581200(4);
    sub_2FF0E80(a1, v25, 0);
  }
  v10 = *(_QWORD *)(a1 + 256);
  if ( (*(_BYTE *)(v10 + 879) & 1) != 0 || (_BYTE)qword_50275A8 )
  {
    sub_2FEF140((__int64)&v42, v10);
    if ( v43 )
    {
      if ( LOBYTE(qword_4F813A8[8]) )
      {
        v18 = *(_QWORD *)(a1 + 256);
        v41 = 0;
        sub_2FEF0A0((__int64)v45, v18);
        v47[0] = (__int64)v48;
        sub_2FEEBD0(v47, v42, (__int64)&v42[v43]);
        v19 = (_QWORD *)sub_3585380(v47, v45, 4, &v41);
        sub_2FF0E80(a1, v19, 0);
        if ( (_QWORD *)v47[0] != v48 )
          j_j___libc_free_0(v47[0]);
        if ( (_QWORD *)v45[0] != v46 )
          j_j___libc_free_0(v45[0]);
        v20 = v41;
        if ( v41 && !_InterlockedSub((volatile signed __int32 *)(v41 + 8), 1u) )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
      }
      else
      {
        v36 = sub_CA5BD0((__int64)&v42, v10);
        v37 = *(__m128i **)(v36 + 32);
        if ( *(_QWORD *)(v36 + 24) - (_QWORD)v37 <= 0x46u )
        {
          sub_CB6200(v36, "Using AutoFDO without FSDiscriminator for MFS may regress performance.\n", 0x47u);
        }
        else
        {
          si128 = _mm_load_si128((const __m128i *)&xmmword_4456EE0);
          v37[4].m128i_i32[0] = 1668178285;
          v37[4].m128i_i16[2] = 11877;
          *v37 = si128;
          v39 = _mm_load_si128((const __m128i *)&xmmword_4456EF0);
          v37[4].m128i_i8[6] = 10;
          v37[1] = v39;
          v37[2] = _mm_load_si128((const __m128i *)&xmmword_4456F00);
          v37[3] = _mm_load_si128((const __m128i *)&xmmword_4456F10);
          *(_QWORD *)(v36 + 32) += 71LL;
        }
      }
    }
    v21 = (_QWORD *)sub_3530020();
    sub_2FF0E80(a1, v21, 0);
    if ( (_BYTE)qword_5027228 || (*(_BYTE *)(*(_QWORD *)(a1 + 256) + 879LL) & 2) != 0 )
    {
      v22 = (_QWORD *)sub_35D4250();
      sub_2FF0E80(a1, v22, 0);
    }
    if ( v42 != (_BYTE *)v44 )
      j_j___libc_free_0((unsigned __int64)v42);
    v10 = *(_QWORD *)(a1 + 256);
    v11 = *(_DWORD *)(v10 + 880);
    if ( v11 != 3 )
      goto LABEL_42;
  }
  else
  {
    v11 = *(_DWORD *)(v10 + 880);
    if ( v11 != 3 )
    {
LABEL_42:
      if ( v11 == 1 )
      {
        v34 = sub_2D514B0(*(_QWORD *)(v10 + 888));
        sub_2FF0E80(a1, v34, 0);
        v35 = (_QWORD *)sub_34BA9A0();
        sub_2FF0E80(a1, v35, 0);
      }
      goto LABEL_44;
    }
  }
  if ( (*(_BYTE *)(v10 + 879) & 0x10) != 0 )
  {
LABEL_44:
    v12 = (_QWORD *)sub_34BC860();
    sub_2FF0E80(a1, v12, 0);
  }
  v13 = *(void (**)())(*(_QWORD *)a1 + 408LL);
  if ( v13 != nullsub_1697 )
    ((void (__fastcall *)(__int64))v13)(a1);
  if ( !(_BYTE)qword_50285E8 && (*(_BYTE *)(*(_QWORD *)(a1 + 256) + 905LL) & 4) != 0 )
  {
    v40 = (_QWORD *)sub_34C9E50();
    sub_2FF0E80(a1, v40, 0);
  }
  v14 = *(_QWORD *)(a1 + 176);
  v15 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v14 + 16LL);
  v16 = sub_35D0760();
  v15(v14, v16, 0);
  result = *(void (**)())(*(_QWORD *)a1 + 416LL);
  if ( result != nullsub_1698 )
    result = (void (*)())((__int64 (__fastcall *)(__int64))result)(a1);
  *(_BYTE *)(a1 + 250) = 0;
  return result;
}

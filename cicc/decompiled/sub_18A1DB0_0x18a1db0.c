// Function: sub_18A1DB0
// Address: 0x18a1db0
//
__int64 __fastcall sub_18A1DB0(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        __m128i a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // eax
  unsigned int v11; // r13d
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // rdx
  void (__fastcall *v28)(_QWORD *, _QWORD *, int); // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+8h] [rbp-B8h] BYREF
  _QWORD v34[2]; // [rsp+10h] [rbp-B0h] BYREF
  __int64 (__fastcall *v35)(_QWORD *, _QWORD *, int); // [rsp+20h] [rbp-A0h]
  __int64 (__fastcall *v36)(__int64 **, __int64); // [rsp+28h] [rbp-98h]
  _QWORD v37[2]; // [rsp+30h] [rbp-90h] BYREF
  void (__fastcall *v38)(_QWORD *, _QWORD *, int); // [rsp+40h] [rbp-80h]
  __int64 (__fastcall *v39)(__int64 **, __int64); // [rsp+48h] [rbp-78h]
  int v40; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v41; // [rsp+58h] [rbp-68h]
  _QWORD *v42; // [rsp+60h] [rbp-60h]
  char v43; // [rsp+78h] [rbp-48h]
  __int64 v44; // [rsp+80h] [rbp-40h]

  v10 = sub_1636800(a1, a2);
  if ( (_BYTE)v10 )
  {
    return 0;
  }
  else
  {
    v13 = *(__int64 **)(a1 + 8);
    v11 = v10;
    v14 = *v13;
    v15 = v13[1];
    if ( v14 == v15 )
LABEL_26:
      BUG();
    while ( *(_UNKNOWN **)v14 != &unk_4F9D764 )
    {
      v14 += 16;
      if ( v15 == v14 )
        goto LABEL_26;
    }
    v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(
            *(_QWORD *)(v14 + 8),
            &unk_4F9D764);
    v17 = *(__int64 **)(a1 + 8);
    v32 = v16;
    v18 = *v17;
    v19 = v17[1];
    if ( v18 == v19 )
LABEL_27:
      BUG();
    while ( *(_UNKNOWN **)v18 != &unk_4F9D3C0 )
    {
      v18 += 16;
      if ( v19 == v18 )
        goto LABEL_27;
    }
    v20 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v18 + 8) + 104LL))(
            *(_QWORD *)(v18 + 8),
            &unk_4F9D3C0);
    v21 = *(__int64 **)(a1 + 8);
    v33 = v20;
    v22 = *v21;
    v23 = v21[1];
    if ( v22 == v23 )
LABEL_28:
      BUG();
    while ( *(_UNKNOWN **)v22 != &unk_4F99CCD )
    {
      v22 += 16;
      if ( v23 == v22 )
        goto LABEL_28;
    }
    v24 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v22 + 8) + 104LL))(
            *(_QWORD *)(v22 + 8),
            &unk_4F99CCD);
    v40 = 0;
    v27 = *(_QWORD *)(v24 + 160);
    v41 = v34;
    v34[0] = &v32;
    v36 = sub_1896F40;
    v35 = sub_1896FE0;
    v37[0] = &v33;
    v39 = sub_1896F50;
    v28 = (void (__fastcall *)(_QWORD *, _QWORD *, int))sub_1897010;
    v38 = (void (__fastcall *)(_QWORD *, _QWORD *, int))sub_1897010;
    v42 = v37;
    v43 = 0;
    v44 = v27;
    if ( byte_4FAD1E0
      || (v11 = sub_18A1B40((__int64)&v40, (__int64)a2, a3, a4, a5, a6, v25, v26, a9, a10), (v28 = v38) != 0) )
    {
      v28(v37, v37, 3);
    }
    if ( v35 )
      ((void (__fastcall *)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64, __int64, __int64))v35)(
        v34,
        v34,
        3,
        v29,
        v30,
        v31,
        v32,
        v33);
  }
  return v11;
}

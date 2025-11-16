// Function: sub_25600F0
// Address: 0x25600f0
//
void __fastcall sub_25600F0(_QWORD *a1, __int64 a2, __int8 *a3, size_t a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // r8
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp-558h] [rbp-558h]
  __int64 v23; // [rsp-558h] [rbp-558h]
  __int64 *v24; // [rsp-550h] [rbp-550h]
  void *v25; // [rsp-548h] [rbp-548h] BYREF
  int v26; // [rsp-540h] [rbp-540h]
  char v27; // [rsp-53Ch] [rbp-53Ch]
  __int64 v28; // [rsp-538h] [rbp-538h]
  __m128i v29; // [rsp-530h] [rbp-530h]
  __int64 v30; // [rsp-520h] [rbp-520h]
  __m128i v31; // [rsp-518h] [rbp-518h]
  __m128i v32; // [rsp-508h] [rbp-508h]
  _QWORD v33[2]; // [rsp-4F8h] [rbp-4F8h] BYREF
  _BYTE v34[324]; // [rsp-4E8h] [rbp-4E8h] BYREF
  int v35; // [rsp-3A4h] [rbp-3A4h]
  __int64 v36; // [rsp-3A0h] [rbp-3A0h]
  _QWORD v37[10]; // [rsp-398h] [rbp-398h] BYREF
  _BYTE v38[352]; // [rsp-348h] [rbp-348h] BYREF
  void *v39; // [rsp-1E8h] [rbp-1E8h] BYREF
  int v40; // [rsp-1E0h] [rbp-1E0h]
  char v41; // [rsp-1DCh] [rbp-1DCh]
  __int64 v42; // [rsp-1D8h] [rbp-1D8h]
  __m128i v43; // [rsp-1D0h] [rbp-1D0h] BYREF
  __int64 v44; // [rsp-1C0h] [rbp-1C0h]
  __m128i v45; // [rsp-1B8h] [rbp-1B8h] BYREF
  __m128i v46; // [rsp-1A8h] [rbp-1A8h] BYREF
  _DWORD v47[84]; // [rsp-198h] [rbp-198h] BYREF
  char v48; // [rsp-48h] [rbp-48h]
  int v49; // [rsp-44h] [rbp-44h]
  __int64 v50; // [rsp-40h] [rbp-40h]

  if ( !a1[549] )
    return;
  v8 = sub_B43CB0(a2);
  v9 = (__int64 *)((__int64 (__fastcall *)(_QWORD, __int64))a1[549])(a1[550], v8);
  v24 = v9;
  if ( a4 <= 2 )
  {
    v10 = *v9;
LABEL_4:
    v22 = v10;
    v11 = sub_B2BE50(v10);
    if ( sub_B6EA50(v11)
      || (v18 = sub_B2BE50(v22),
          v19 = sub_B6F970(v18),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v19 + 48LL))(v19)) )
    {
      sub_B174A0((__int64)&v39, a1[551], (__int64)a3, a4, a2);
      sub_255FEC0((__int64)v37, *(__int64 ***)a5, *(__int64 **)(a5 + 8), (__int64)&v39);
      v39 = &unk_49D9D40;
      sub_23FD590((__int64)v47);
      sub_1049740(v24, (__int64)v37);
      v37[0] = &unk_49D9D40;
      sub_23FD590((__int64)v38);
    }
    return;
  }
  if ( *(_WORD *)a3 != 19791 || (v12 = 0, a3[2] != 80) )
    v12 = 1;
  v10 = *v24;
  if ( v12 )
    goto LABEL_4;
  v23 = *v24;
  v13 = sub_B2BE50(*v24);
  if ( sub_B6EA50(v13)
    || (v20 = sub_B2BE50(v23),
        v21 = sub_B6F970(v20),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v21 + 48LL))(v21)) )
  {
    sub_B174A0((__int64)v37, a1[551], (__int64)a3, a4, a2);
    sub_255FEC0((__int64)&v39, *(__int64 ***)a5, *(__int64 **)(a5 + 8), (__int64)v37);
    sub_B18290((__int64)&v39, " [", 2u);
    sub_B18290((__int64)&v39, a3, a4);
    sub_B18290((__int64)&v39, "]", 1u);
    v26 = v40;
    v29 = _mm_loadu_si128(&v43);
    v27 = v41;
    v31 = _mm_loadu_si128(&v45);
    v28 = v42;
    v25 = &unk_49D9D40;
    v32 = _mm_loadu_si128(&v46);
    v30 = v44;
    v33[0] = v34;
    v33[1] = 0x400000000LL;
    if ( v47[2] )
      sub_255FC40((__int64)v33, (__int64)v47, v14, v15, v16, v17);
    v39 = &unk_49D9D40;
    v34[320] = v48;
    v35 = v49;
    v36 = v50;
    v25 = &unk_49D9D78;
    sub_23FD590((__int64)v47);
    v37[0] = &unk_49D9D40;
    sub_23FD590((__int64)v38);
    sub_1049740(v24, (__int64)&v25);
    v25 = &unk_49D9D40;
    sub_23FD590((__int64)v33);
  }
}

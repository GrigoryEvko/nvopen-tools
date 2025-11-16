// Function: sub_2560480
// Address: 0x2560480
//
void __fastcall sub_2560480(_QWORD *a1, __int64 a2, __int8 *a3, size_t a4)
{
  __int64 v6; // rax
  __int64 *v7; // rax
  __int64 *v8; // r15
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r9
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r9
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __m128i v20; // xmm2
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // [rsp-550h] [rbp-550h]
  __int64 v28; // [rsp-550h] [rbp-550h]
  void *v29; // [rsp-548h] [rbp-548h] BYREF
  int v30; // [rsp-540h] [rbp-540h]
  char v31; // [rsp-53Ch] [rbp-53Ch]
  __int64 v32; // [rsp-538h] [rbp-538h]
  __m128i v33; // [rsp-530h] [rbp-530h]
  __int64 v34; // [rsp-520h] [rbp-520h]
  __m128i v35; // [rsp-518h] [rbp-518h]
  __m128i v36; // [rsp-508h] [rbp-508h]
  _QWORD v37[2]; // [rsp-4F8h] [rbp-4F8h] BYREF
  _BYTE v38[324]; // [rsp-4E8h] [rbp-4E8h] BYREF
  int v39; // [rsp-3A4h] [rbp-3A4h]
  __int64 v40; // [rsp-3A0h] [rbp-3A0h]
  void *v41; // [rsp-398h] [rbp-398h] BYREF
  int v42; // [rsp-390h] [rbp-390h]
  char v43; // [rsp-38Ch] [rbp-38Ch]
  __int64 v44; // [rsp-388h] [rbp-388h]
  __m128i v45; // [rsp-380h] [rbp-380h] BYREF
  __int64 v46; // [rsp-370h] [rbp-370h]
  __m128i v47; // [rsp-368h] [rbp-368h] BYREF
  __m128i v48; // [rsp-358h] [rbp-358h] BYREF
  __int64 *v49; // [rsp-348h] [rbp-348h] BYREF
  __int64 v50; // [rsp-340h] [rbp-340h]
  __int64 v51; // [rsp-338h] [rbp-338h] BYREF
  char v52; // [rsp-1F8h] [rbp-1F8h]
  int v53; // [rsp-1F4h] [rbp-1F4h]
  __int64 v54; // [rsp-1F0h] [rbp-1F0h]
  void *v55; // [rsp-1E8h] [rbp-1E8h] BYREF
  int v56; // [rsp-1E0h] [rbp-1E0h]
  char v57; // [rsp-1DCh] [rbp-1DCh]
  __int64 v58; // [rsp-1D8h] [rbp-1D8h]
  __m128i v59; // [rsp-1D0h] [rbp-1D0h] BYREF
  __int64 v60; // [rsp-1C0h] [rbp-1C0h]
  __m128i v61; // [rsp-1B8h] [rbp-1B8h] BYREF
  __m128i v62; // [rsp-1A8h] [rbp-1A8h] BYREF
  __int64 *v63; // [rsp-198h] [rbp-198h] BYREF
  __int64 v64; // [rsp-190h] [rbp-190h]
  __int64 v65; // [rsp-188h] [rbp-188h] BYREF
  char v66; // [rsp-48h] [rbp-48h]
  int v67; // [rsp-44h] [rbp-44h]
  __int64 v68; // [rsp-40h] [rbp-40h]

  if ( !a1[549] )
    return;
  v6 = sub_B43CB0(a2);
  v7 = (__int64 *)((__int64 (__fastcall *)(_QWORD, __int64))a1[549])(a1[550], v6);
  v8 = v7;
  if ( a4 <= 2 )
  {
    v9 = *v7;
LABEL_4:
    v27 = v9;
    v10 = sub_B2BE50(v9);
    if ( sub_B6EA50(v10)
      || (v23 = sub_B2BE50(v27),
          v24 = sub_B6F970(v23),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v24 + 48LL))(v24)) )
    {
      sub_B176B0((__int64)&v55, a1[551], (__int64)a3, a4, a2);
      sub_B18290(
        (__int64)&v55,
        "Could not move globalized variable to the stack. Variable is potentially captured in call. Mark parameter as `__"
        "attribute__((noescape))` to override.",
        0x95u);
      v42 = v56;
      v45 = _mm_loadu_si128(&v59);
      v43 = v57;
      v47 = _mm_loadu_si128(&v61);
      v44 = v58;
      v41 = &unk_49D9D40;
      v48 = _mm_loadu_si128(&v62);
      v46 = v60;
      v49 = &v51;
      v50 = 0x400000000LL;
      if ( (_DWORD)v64 )
        sub_255FC40((__int64)&v49, (__int64)&v63, v11, v12, (__int64)&v63, v13);
      v55 = &unk_49D9D40;
      v52 = v66;
      v53 = v67;
      v54 = v68;
      v41 = &unk_49D9DB0;
      sub_23FD590((__int64)&v63);
      sub_1049740(v8, (__int64)&v41);
      v41 = &unk_49D9D40;
      sub_23FD590((__int64)&v49);
    }
    return;
  }
  if ( *(_WORD *)a3 != 19791 || (v14 = 0, a3[2] != 80) )
    v14 = 1;
  v9 = *v8;
  if ( v14 )
    goto LABEL_4;
  v28 = *v8;
  v15 = sub_B2BE50(*v8);
  if ( sub_B6EA50(v15)
    || (v25 = sub_B2BE50(v28),
        v26 = sub_B6F970(v25),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v26 + 48LL))(v26)) )
  {
    sub_B176B0((__int64)&v41, a1[551], (__int64)a3, a4, a2);
    sub_B18290(
      (__int64)&v41,
      "Could not move globalized variable to the stack. Variable is potentially captured in call. Mark parameter as `__at"
      "tribute__((noescape))` to override.",
      0x95u);
    v18 = _mm_loadu_si128(&v45);
    v19 = _mm_loadu_si128(&v47);
    v56 = v42;
    v20 = _mm_loadu_si128(&v48);
    v59 = v18;
    v57 = v43;
    v61 = v19;
    v58 = v44;
    v55 = &unk_49D9D40;
    v62 = v20;
    v60 = v46;
    v63 = &v65;
    v64 = 0x400000000LL;
    if ( (_DWORD)v50 )
      sub_255FC40((__int64)&v63, (__int64)&v49, v16, (unsigned int)v50, (__int64)&v63, v17);
    v66 = v52;
    v67 = v53;
    v68 = v54;
    v55 = &unk_49D9DB0;
    sub_B18290((__int64)&v55, " [", 2u);
    sub_B18290((__int64)&v55, a3, a4);
    sub_B18290((__int64)&v55, "]", 1u);
    v30 = v56;
    v33 = _mm_loadu_si128(&v59);
    v31 = v57;
    v35 = _mm_loadu_si128(&v61);
    v32 = v58;
    v29 = &unk_49D9D40;
    v36 = _mm_loadu_si128(&v62);
    v34 = v60;
    v37[0] = v38;
    v37[1] = 0x400000000LL;
    if ( (_DWORD)v64 )
      sub_255FC40((__int64)v37, (__int64)&v63, (unsigned int)v64, v21, (__int64)&v63, v22);
    v55 = &unk_49D9D40;
    v38[320] = v66;
    v39 = v67;
    v40 = v68;
    v29 = &unk_49D9DB0;
    sub_23FD590((__int64)&v63);
    v41 = &unk_49D9D40;
    sub_23FD590((__int64)&v49);
    sub_1049740(v8, (__int64)&v29);
    v29 = &unk_49D9D40;
    sub_23FD590((__int64)v37);
  }
}

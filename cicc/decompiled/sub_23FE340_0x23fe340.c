// Function: sub_23FE340
// Address: 0x23fe340
//
void __fastcall sub_23FE340(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rax
  void *v10; // [rsp+0h] [rbp-380h] BYREF
  int v11; // [rsp+8h] [rbp-378h]
  char v12; // [rsp+Ch] [rbp-374h]
  __int64 v13; // [rsp+10h] [rbp-370h]
  __m128i v14; // [rsp+18h] [rbp-368h]
  __int64 v15; // [rsp+28h] [rbp-358h]
  __m128i v16; // [rsp+30h] [rbp-350h]
  __m128i v17; // [rsp+40h] [rbp-340h]
  _QWORD v18[2]; // [rsp+50h] [rbp-330h] BYREF
  _BYTE v19[324]; // [rsp+60h] [rbp-320h] BYREF
  int v20; // [rsp+1A4h] [rbp-1DCh]
  __int64 v21; // [rsp+1A8h] [rbp-1D8h]
  void *v22; // [rsp+1B0h] [rbp-1D0h] BYREF
  int v23; // [rsp+1B8h] [rbp-1C8h]
  char v24; // [rsp+1BCh] [rbp-1C4h]
  __int64 v25; // [rsp+1C0h] [rbp-1C0h]
  __m128i v26; // [rsp+1C8h] [rbp-1B8h] BYREF
  __int64 v27; // [rsp+1D8h] [rbp-1A8h]
  __m128i v28; // [rsp+1E0h] [rbp-1A0h] BYREF
  __m128i v29; // [rsp+1F0h] [rbp-190h] BYREF
  _BYTE v30[8]; // [rsp+200h] [rbp-180h] BYREF
  int v31; // [rsp+208h] [rbp-178h]
  char v32; // [rsp+350h] [rbp-30h]
  int v33; // [rsp+354h] [rbp-2Ch]
  __int64 v34; // [rsp+358h] [rbp-28h]

  v2 = *a1;
  v3 = sub_B2BE50(*a1);
  if ( sub_B6EA50(v3)
    || (v8 = sub_B2BE50(v2), v9 = sub_B6F970(v8),
                             (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v9 + 48LL))(v9)) )
  {
    sub_B176B0((__int64)&v22, (__int64)"chr", (__int64)"SelectNotBiased", 15, *a2);
    sub_B18290((__int64)&v22, "Select not biased", 0x11u);
    v11 = v23;
    v14 = _mm_loadu_si128(&v26);
    v12 = v24;
    v16 = _mm_loadu_si128(&v28);
    v13 = v25;
    v10 = &unk_49D9D40;
    v17 = _mm_loadu_si128(&v29);
    v15 = v27;
    v18[0] = v19;
    v18[1] = 0x400000000LL;
    if ( v31 )
      sub_23FE010((__int64)v18, (__int64)v30, v4, v5, v6, v7);
    v22 = &unk_49D9D40;
    v19[320] = v32;
    v20 = v33;
    v21 = v34;
    v10 = &unk_49D9DB0;
    sub_23FD590((__int64)v30);
    sub_1049740(a1, (__int64)&v10);
    v10 = &unk_49D9D40;
    sub_23FD590((__int64)v18);
  }
}

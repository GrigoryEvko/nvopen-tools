// Function: sub_3752FF0
// Address: 0x3752ff0
//
__int64 __fastcall sub_3752FF0(
        __int64 *a1,
        unsigned int a2,
        unsigned int a3,
        unsigned __int16 a4,
        unsigned __int8 a5,
        unsigned __int8 **a6)
{
  __int64 v9; // r12
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned __int64 v13; // rcx
  __int64 (__fastcall *v14)(__int64, __int64); // rax
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r9
  __int64 v19; // rdi
  __int64 (__fastcall *v20)(__int64, __int64); // r8
  __int64 (__fastcall *v21)(__int64, unsigned __int16); // rax
  __int64 v22; // rsi
  unsigned __int32 v23; // eax
  unsigned __int8 *v24; // rsi
  unsigned __int32 v25; // r12d
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 *v28; // rsi
  __int64 v29; // rdi
  _QWORD *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 (__fastcall *v33)(__int64, __int64); // [rsp+8h] [rbp-A8h]
  unsigned __int64 v34; // [rsp+10h] [rbp-A0h]
  __int64 v35; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v37; // [rsp+28h] [rbp-88h] BYREF
  __int64 v38[4]; // [rsp+30h] [rbp-80h] BYREF
  __m128i v39; // [rsp+50h] [rbp-60h] BYREF
  __int64 v40; // [rsp+60h] [rbp-50h]
  __int64 v41; // [rsp+68h] [rbp-48h]
  __int64 v42; // [rsp+70h] [rbp-40h]

  v9 = a4;
  v11 = a1[1];
  v12 = a1[3];
  v13 = *(_QWORD *)(*(_QWORD *)(v11 + 56) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v14 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v12 + 272LL);
  if ( v14 != sub_2E85430 )
  {
    v34 = *(_QWORD *)(*(_QWORD *)(v11 + 56) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v16 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v14)(v12, v13, a3);
    if ( v16 && v34 != v16 )
    {
      v17 = sub_2EBE590(a1[1], a2, v16, 4u);
      result = a2;
      if ( v17 )
        return result;
      goto LABEL_7;
    }
    v13 = v16;
  }
  result = a2;
  if ( v13 )
    return result;
LABEL_7:
  v18 = a1[3];
  v19 = a1[4];
  v20 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v18 + 272LL);
  v21 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v19 + 552LL);
  if ( v21 == sub_2EC09E0 )
  {
    v22 = *(_QWORD *)(v19 + 8 * v9 + 112);
  }
  else
  {
    v33 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v18 + 272LL);
    v35 = a1[3];
    v32 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v21)(v19, (unsigned int)v9, a5);
    v20 = v33;
    v18 = v35;
    v22 = v32;
  }
  if ( v20 != sub_2E85430 )
    v22 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v20)(v18, v22, a3);
  v23 = sub_2EC06C0(a1[1], v22, byte_3F871B3, 0, (__int64)v20, v18);
  v24 = *a6;
  v25 = v23;
  v26 = a1[2];
  v37 = v24;
  v27 = *(_QWORD *)(v26 + 8) - 800LL;
  if ( v24 )
  {
    sub_B96E90((__int64)&v37, (__int64)v24, 1);
    v38[0] = (__int64)v37;
    if ( v37 )
    {
      sub_B976B0((__int64)&v37, v37, (__int64)v38);
      v37 = 0;
    }
  }
  else
  {
    v38[0] = 0;
  }
  v28 = (__int64 *)a1[6];
  v29 = a1[5];
  v38[1] = 0;
  v38[2] = 0;
  v30 = sub_2F26260(v29, v28, v38, v27, v25);
  v39.m128i_i64[0] = 0;
  v40 = 0;
  v39.m128i_i32[2] = a2;
  v41 = 0;
  v42 = 0;
  sub_2E8EAD0(v31, (__int64)v30, &v39);
  if ( v38[0] )
    sub_B91220((__int64)v38, v38[0]);
  if ( v37 )
    sub_B91220((__int64)&v37, (__int64)v37);
  return v25;
}

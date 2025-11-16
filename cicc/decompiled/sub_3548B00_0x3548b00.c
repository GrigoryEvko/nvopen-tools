// Function: sub_3548B00
// Address: 0x3548b00
//
void __fastcall sub_3548B00(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __m128i v4; // xmm0
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r9
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // [rsp+18h] [rbp-3A8h] BYREF
  __m128i v15; // [rsp+20h] [rbp-3A0h] BYREF
  void *v16; // [rsp+30h] [rbp-390h] BYREF
  int v17; // [rsp+38h] [rbp-388h]
  char v18; // [rsp+3Ch] [rbp-384h]
  __int64 v19; // [rsp+40h] [rbp-380h]
  __m128i v20; // [rsp+48h] [rbp-378h]
  char *v21; // [rsp+58h] [rbp-368h]
  __m128i v22; // [rsp+60h] [rbp-360h]
  __m128i v23; // [rsp+70h] [rbp-350h]
  _QWORD v24[2]; // [rsp+80h] [rbp-340h] BYREF
  _BYTE v25[324]; // [rsp+90h] [rbp-330h] BYREF
  int v26; // [rsp+1D4h] [rbp-1ECh]
  __int64 v27; // [rsp+1D8h] [rbp-1E8h]
  void *v28; // [rsp+1E0h] [rbp-1E0h] BYREF
  __int64 v29; // [rsp+1E8h] [rbp-1D8h]
  __int64 v30; // [rsp+1F0h] [rbp-1D0h]
  __m128i v31; // [rsp+1F8h] [rbp-1C8h] BYREF
  char *v32; // [rsp+208h] [rbp-1B8h]
  __m128i v33; // [rsp+210h] [rbp-1B0h] BYREF
  __m128i v34; // [rsp+220h] [rbp-1A0h] BYREF
  char *v35; // [rsp+230h] [rbp-190h] BYREF
  __int64 v36; // [rsp+238h] [rbp-188h]
  char v37; // [rsp+240h] [rbp-180h] BYREF
  char v38; // [rsp+380h] [rbp-40h]
  int v39; // [rsp+384h] [rbp-3Ch]
  __int64 v40; // [rsp+388h] [rbp-38h]

  v2 = sub_B2BE50(**a1);
  if ( sub_B6EA50(v2)
    || (v12 = sub_B2BE50(**a1),
        v13 = sub_B6F970(v12),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v13 + 48LL))(v13)) )
  {
    v3 = **(_QWORD **)(a2 + 32);
    sub_2EA6600(&v14, a2);
    sub_B157E0((__int64)&v15, &v14);
    v4 = _mm_loadu_si128(&v15);
    v5 = **(_QWORD **)(v3 + 32);
    v36 = 0x400000000LL;
    v40 = v3;
    v30 = v5;
    v32 = "pipeliner";
    v33.m128i_i64[0] = (__int64)"canPipelineLoop";
    v35 = &v37;
    v28 = &unk_4A28EB8;
    v29 = 0x200000015LL;
    v33.m128i_i64[1] = 15;
    v34.m128i_i8[8] = 0;
    v38 = 0;
    v39 = -1;
    v31 = v4;
    sub_B18290((__int64)&v28, "The loop structure is not supported", 0x23u);
    v24[1] = 0x400000000LL;
    v9 = _mm_loadu_si128(&v31);
    v10 = _mm_loadu_si128(&v33);
    v17 = v29;
    v11 = _mm_loadu_si128(&v34);
    v20 = v9;
    v18 = BYTE4(v29);
    v22 = v10;
    v19 = v30;
    v16 = &unk_49D9D40;
    v23 = v11;
    v21 = v32;
    v24[0] = v25;
    if ( (_DWORD)v36 )
      sub_35482E0((__int64)v24, (__int64)&v35, v6, v7, (__int64)&v35, v8);
    v25[320] = v38;
    v26 = v39;
    v27 = v40;
    v16 = &unk_4A28EB8;
    v28 = &unk_49D9D40;
    sub_23FD590((__int64)&v35);
    if ( v14 )
      sub_B91220((__int64)&v14, v14);
    sub_2EAFC50(a1, (__int64)&v16);
    v16 = &unk_49D9D40;
    sub_23FD590((__int64)v24);
  }
}

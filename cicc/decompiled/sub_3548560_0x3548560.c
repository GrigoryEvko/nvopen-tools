// Function: sub_3548560
// Address: 0x3548560
//
void __fastcall sub_3548560(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __m128i v4; // xmm0
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-408h]
  __int64 v21; // [rsp+18h] [rbp-3F8h] BYREF
  __m128i v22; // [rsp+20h] [rbp-3F0h] BYREF
  __int64 v23[2]; // [rsp+30h] [rbp-3E0h] BYREF
  __int64 v24; // [rsp+40h] [rbp-3D0h] BYREF
  __int64 *v25; // [rsp+50h] [rbp-3C0h]
  __int64 v26; // [rsp+60h] [rbp-3B0h] BYREF
  void *v27; // [rsp+80h] [rbp-390h] BYREF
  int v28; // [rsp+88h] [rbp-388h]
  char v29; // [rsp+8Ch] [rbp-384h]
  __int64 v30; // [rsp+90h] [rbp-380h]
  __m128i v31; // [rsp+98h] [rbp-378h]
  __int64 v32; // [rsp+A8h] [rbp-368h]
  __m128i v33; // [rsp+B0h] [rbp-360h]
  __m128i v34; // [rsp+C0h] [rbp-350h]
  _QWORD v35[2]; // [rsp+D0h] [rbp-340h] BYREF
  _BYTE v36[324]; // [rsp+E0h] [rbp-330h] BYREF
  int v37; // [rsp+224h] [rbp-1ECh]
  __int64 v38; // [rsp+228h] [rbp-1E8h]
  _QWORD v39[3]; // [rsp+230h] [rbp-1E0h] BYREF
  __m128i v40; // [rsp+248h] [rbp-1C8h]
  char *v41; // [rsp+258h] [rbp-1B8h]
  const char *v42; // [rsp+260h] [rbp-1B0h]
  __int64 v43; // [rsp+268h] [rbp-1A8h]
  char v44; // [rsp+278h] [rbp-198h]
  _QWORD v45[2]; // [rsp+280h] [rbp-190h] BYREF
  _BYTE v46[324]; // [rsp+290h] [rbp-180h] BYREF
  int v47; // [rsp+3D4h] [rbp-3Ch]
  __int64 v48; // [rsp+3D8h] [rbp-38h]

  v2 = sub_B2BE50(**a1);
  if ( sub_B6EA50(v2)
    || (v18 = sub_B2BE50(**a1),
        v19 = sub_B6F970(v18),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v19 + 48LL))(v19)) )
  {
    v3 = **(_QWORD **)(a2 + 32);
    sub_2EA6600(&v21, a2);
    sub_B157E0((__int64)&v22, &v21);
    v4 = _mm_loadu_si128(&v22);
    v5 = **(_QWORD **)(v3 + 32);
    v48 = v3;
    v39[2] = v5;
    v41 = "pipeliner";
    v42 = "canPipelineLoop";
    v45[0] = v46;
    v39[0] = &unk_4A28EB8;
    v40 = v4;
    v39[1] = 0x200000015LL;
    v45[1] = 0x400000000LL;
    v43 = 15;
    v44 = 0;
    v46[320] = 0;
    v47 = -1;
    sub_B18290((__int64)v39, "Not a single basic block: ", 0x1Au);
    sub_B169E0(v23, "NumBlocks", 9, (__int64)(*(_QWORD *)(a2 + 40) - *(_QWORD *)(a2 + 32)) >> 3);
    v6 = sub_2E82FF0((__int64)v39, (__int64)v23);
    v10 = _mm_loadu_si128((const __m128i *)(v6 + 24));
    v11 = _mm_loadu_si128((const __m128i *)(v6 + 48));
    v12 = _mm_loadu_si128((const __m128i *)(v6 + 64));
    v28 = *(_DWORD *)(v6 + 8);
    v29 = *(_BYTE *)(v6 + 12);
    v13 = *(_QWORD *)(v6 + 16);
    v31 = v10;
    v30 = v13;
    v27 = &unk_49D9D40;
    v14 = *(_QWORD *)(v6 + 40);
    v35[1] = 0x400000000LL;
    v32 = v14;
    v35[0] = v36;
    v15 = *(unsigned int *)(v6 + 88);
    v33 = v11;
    v34 = v12;
    if ( (_DWORD)v15 )
    {
      v20 = v6;
      sub_35482E0((__int64)v35, v6 + 80, v15, v7, v8, v9);
      v6 = v20;
    }
    v36[320] = *(_BYTE *)(v6 + 416);
    v16 = *(_DWORD *)(v6 + 420);
    v17 = *(_QWORD *)(v6 + 424);
    v37 = v16;
    v38 = v17;
    v27 = &unk_4A28EB8;
    if ( v25 != &v26 )
      j_j___libc_free_0((unsigned __int64)v25);
    if ( (__int64 *)v23[0] != &v24 )
      j_j___libc_free_0(v23[0]);
    v39[0] = &unk_49D9D40;
    sub_23FD590((__int64)v45);
    if ( v21 )
      sub_B91220((__int64)&v21, v21);
    sub_2EAFC50(a1, (__int64)&v27);
    v27 = &unk_49D9D40;
    sub_23FD590((__int64)v35);
  }
}

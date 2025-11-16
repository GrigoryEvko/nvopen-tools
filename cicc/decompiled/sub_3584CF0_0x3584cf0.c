// Function: sub_3584CF0
// Address: 0x3584cf0
//
void __fastcall sub_3584CF0(__int64 a1, __int64 a2, __int64 *a3, int a4, volatile signed __int32 **a5)
{
  __int64 v10; // rax
  __int64 *v11; // rdi
  int v12; // eax
  volatile signed __int32 *v13; // r10
  __int64 v14; // rsi
  _BYTE *v15; // r14
  __int64 v16; // r13
  __int64 v17; // r12
  __int64 v18; // rax
  volatile signed __int32 *v19; // r10
  __int64 v20; // rbx
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rdx
  _QWORD *v24; // rdi
  _QWORD *v25; // r12
  volatile signed __int32 *v26; // rdi
  volatile signed __int32 *v27; // r10
  __int64 v28; // rax
  volatile signed __int32 *v29; // [rsp+0h] [rbp-90h]
  volatile signed __int32 *v30; // [rsp+0h] [rbp-90h]
  volatile signed __int32 *v31; // [rsp+8h] [rbp-88h]
  volatile signed __int32 *v32; // [rsp+18h] [rbp-78h] BYREF
  __int64 v33[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v34[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v35[2]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v36[8]; // [rsp+50h] [rbp-40h] BYREF

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 16) = &unk_503F254;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  v10 = a1 + 160;
  v11 = (__int64 *)(a1 + 208);
  *(v11 - 12) = v10;
  *(v11 - 22) = 0;
  *(v11 - 21) = 0;
  *(v11 - 20) = 0;
  *(v11 - 18) = 1;
  *(v11 - 17) = 0;
  *(v11 - 16) = 0;
  *((_DWORD *)v11 - 30) = 1065353216;
  *(v11 - 14) = 0;
  *(v11 - 13) = 0;
  *(v11 - 11) = 1;
  *(v11 - 10) = 0;
  *(v11 - 9) = 0;
  *((_DWORD *)v11 - 16) = 1065353216;
  *(v11 - 7) = 0;
  *(v11 - 6) = 0;
  *((_BYTE *)v11 - 40) = 0;
  *(v11 - 4) = 0;
  *(v11 - 3) = 0;
  *(v11 - 2) = 0;
  *(v11 - 26) = (__int64)&unk_4A39928;
  *(_QWORD *)(a1 + 208) = a1 + 224;
  sub_3583BC0(v11, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  *(_DWORD *)(a1 + 240) = a4;
  v12 = 0;
  *(_QWORD *)(a1 + 256) = 0;
  if ( a4 )
    v12 = 2 * (3 * a4 - 3) + 8;
  v13 = *a5;
  *(_DWORD *)(a1 + 244) = v12;
  *(_DWORD *)(a1 + 248) = 6 * a4 + 7;
  v31 = v13;
  if ( !v13 )
  {
    sub_CA41E0(&v32);
    v27 = v32;
    v14 = *a3;
    v15 = *(_BYTE **)a2;
    v16 = a3[1];
    v32 = 0;
    v30 = v27;
    v17 = *(_QWORD *)(a2 + 8);
    v28 = sub_22077B0(0x528u);
    v19 = v30;
    v20 = v28;
    if ( v28 )
      goto LABEL_5;
    if ( !v30 )
    {
LABEL_18:
      v20 = 0;
      goto LABEL_10;
    }
LABEL_17:
    if ( !_InterlockedSub(v19 + 2, 1u) )
    {
      v20 = 0;
      (*(void (__fastcall **)(volatile signed __int32 *, __int64))(*(_QWORD *)v19 + 8LL))(v19, v14);
      goto LABEL_10;
    }
    goto LABEL_18;
  }
  *a5 = 0;
  v14 = *a3;
  v15 = *(_BYTE **)a2;
  v16 = a3[1];
  v32 = 0;
  v17 = *(_QWORD *)(a2 + 8);
  v18 = sub_22077B0(0x528u);
  v19 = v31;
  v20 = v18;
  if ( !v18 )
    goto LABEL_17;
LABEL_5:
  v29 = v19;
  v35[0] = (__int64)v36;
  sub_3583D30(v35, (_BYTE *)v14, v14 + v16);
  v33[0] = (__int64)v34;
  sub_3583D30(v33, v15, (__int64)&v15[v17]);
  *(_QWORD *)(v20 + 8) = 0;
  *(_QWORD *)(v20 + 16) = 0;
  *(_QWORD *)(v20 + 24) = 0;
  *(_QWORD *)v20 = &unk_4A398D8;
  *(_QWORD *)(v20 + 112) = v20 + 136;
  *(_QWORD *)(v20 + 392) = v20 + 408;
  *(_QWORD *)(v20 + 400) = 0x2000000000LL;
  *(_QWORD *)(v20 + 944) = v20 + 928;
  *(_QWORD *)(v20 + 952) = v20 + 928;
  *(_DWORD *)(v20 + 32) = 0;
  *(_QWORD *)(v20 + 40) = 0;
  *(_QWORD *)(v20 + 48) = 0;
  *(_QWORD *)(v20 + 56) = 0;
  *(_DWORD *)(v20 + 64) = 0;
  *(_QWORD *)(v20 + 72) = 0;
  *(_QWORD *)(v20 + 80) = 0;
  *(_QWORD *)(v20 + 88) = 0;
  *(_DWORD *)(v20 + 96) = 0;
  *(_QWORD *)(v20 + 104) = 0;
  *(_QWORD *)(v20 + 120) = 32;
  *(_DWORD *)(v20 + 128) = 0;
  *(_BYTE *)(v20 + 132) = 1;
  *(_DWORD *)(v20 + 928) = 0;
  *(_QWORD *)(v20 + 936) = 0;
  *(_QWORD *)(v20 + 960) = 0;
  *(_QWORD *)(v20 + 968) = 0;
  *(_QWORD *)(v20 + 976) = 0;
  *(_QWORD *)(v20 + 984) = 0;
  *(_DWORD *)(v20 + 992) = 0;
  *(_QWORD *)(v20 + 1024) = 0;
  *(_QWORD *)(v20 + 1032) = 0;
  *(_QWORD *)(v20 + 1040) = 0;
  v21 = (_BYTE *)v33[0];
  *(_QWORD *)(v20 + 1168) = v20 + 1152;
  v22 = v33[1];
  *(_QWORD *)(v20 + 1176) = v20 + 1152;
  *(_QWORD *)(v20 + 1208) = v20 + 1224;
  *(_DWORD *)(v20 + 1048) = 0;
  *(_QWORD *)(v20 + 1056) = 0;
  *(_QWORD *)(v20 + 1064) = 0;
  *(_QWORD *)(v20 + 1072) = 0;
  *(_DWORD *)(v20 + 1080) = 0;
  *(_QWORD *)(v20 + 1088) = 0;
  *(_QWORD *)(v20 + 1096) = 0;
  *(_QWORD *)(v20 + 1104) = 0;
  *(_DWORD *)(v20 + 1112) = 0;
  *(_QWORD *)(v20 + 1120) = 0;
  *(_BYTE *)(v20 + 1128) = 0;
  *(_QWORD *)(v20 + 1136) = 0;
  *(_DWORD *)(v20 + 1152) = 0;
  *(_QWORD *)(v20 + 1160) = 0;
  *(_QWORD *)(v20 + 1184) = 0;
  *(_QWORD *)(v20 + 1192) = 0;
  *(_QWORD *)(v20 + 1200) = 0;
  sub_3583BC0((__int64 *)(v20 + 1208), v21, (__int64)&v21[v22]);
  v14 = v35[0];
  v23 = v35[1];
  *(_QWORD *)(v20 + 1240) = v20 + 1256;
  sub_3583BC0((__int64 *)(v20 + 1240), (_BYTE *)v14, v14 + v23);
  v24 = (_QWORD *)v33[0];
  *(_QWORD *)(v20 + 1280) = 0;
  *(_QWORD *)(v20 + 1288) = 0;
  *(_QWORD *)(v20 + 1272) = v29;
  if ( v24 != v34 )
  {
    v14 = v34[0] + 1LL;
    j_j___libc_free_0((unsigned __int64)v24);
  }
  if ( (_QWORD *)v35[0] != v36 )
  {
    v14 = v36[0] + 1LL;
    j_j___libc_free_0(v35[0]);
  }
  *(_BYTE *)(v20 + 1316) = 1;
  *(_QWORD *)v20 = &unk_4A39900;
LABEL_10:
  v25 = *(_QWORD **)(a1 + 256);
  *(_QWORD *)(a1 + 256) = v20;
  if ( v25 )
  {
    *v25 = &unk_4A39900;
    sub_3584890((__int64)v25);
    v14 = 1320;
    j_j___libc_free_0((unsigned __int64)v25);
  }
  v26 = v32;
  if ( v32 )
  {
    if ( !_InterlockedSub(v32 + 2, 1u) )
      (*(void (__fastcall **)(volatile signed __int32 *, __int64))(*(_QWORD *)v26 + 8LL))(v26, v14);
  }
}

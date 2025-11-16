// Function: sub_2607F90
// Address: 0x2607f90
//
__int64 __fastcall sub_2607F90(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD v14[4]; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v15; // [rsp+20h] [rbp-90h]
  __int64 v16; // [rsp+28h] [rbp-88h]
  unsigned int v17; // [rsp+30h] [rbp-80h]
  __int64 v18; // [rsp+38h] [rbp-78h]
  __int64 v19; // [rsp+40h] [rbp-70h]
  __int64 v20; // [rsp+48h] [rbp-68h]
  unsigned int v21; // [rsp+50h] [rbp-60h]
  __int64 v22; // [rsp+58h] [rbp-58h]
  __int64 v23; // [rsp+60h] [rbp-50h]
  __int64 v24; // [rsp+68h] [rbp-48h]
  unsigned int v25; // [rsp+70h] [rbp-40h]
  __int64 v26; // [rsp+78h] [rbp-38h]
  __int64 v27; // [rsp+80h] [rbp-30h]
  __int64 v28; // [rsp+88h] [rbp-28h]
  unsigned int v29; // [rsp+90h] [rbp-20h]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a1 + 16);
  v14[3] = 1;
  v4 = *(_QWORD *)a1;
  ++*(_QWORD *)(a1 + 24);
  v14[1] = v2;
  v5 = *(_QWORD *)(a1 + 32);
  v14[2] = v3;
  LODWORD(v3) = *(_DWORD *)(a1 + 48);
  v15 = v5;
  v6 = *(_QWORD *)(a1 + 40);
  v17 = v3;
  LODWORD(v3) = *(_DWORD *)(a1 + 80);
  v16 = v6;
  v7 = *(_QWORD *)(a1 + 64);
  v14[0] = v4;
  v19 = v7;
  v8 = *(_QWORD *)(a1 + 72);
  v21 = v3;
  v20 = v8;
  ++*(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  v18 = 1;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 80) = 0;
  v9 = *(_QWORD *)(a1 + 96);
  LODWORD(v3) = *(_DWORD *)(a1 + 112);
  ++*(_QWORD *)(a1 + 88);
  v23 = v9;
  v10 = *(_QWORD *)(a1 + 104);
  ++*(_QWORD *)(a1 + 120);
  v24 = v10;
  v11 = *(_QWORD *)(a1 + 128);
  v25 = v3;
  LODWORD(v3) = *(_DWORD *)(a1 + 144);
  v27 = v11;
  v12 = *(_QWORD *)(a1 + 136);
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  v29 = v3;
  v28 = v12;
  v22 = 1;
  v26 = 1;
  sub_25F6310(a1, a2);
  sub_25F6310(a2, (__int64)v14);
  sub_C7D6A0(v27, 8LL * v29, 4);
  sub_C7D6A0(v23, 8LL * v25, 4);
  sub_C7D6A0(v19, 16LL * v21, 8);
  return sub_C7D6A0(v15, 16LL * v17, 8);
}

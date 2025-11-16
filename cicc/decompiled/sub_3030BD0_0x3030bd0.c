// Function: sub_3030BD0
// Address: 0x3030bd0
//
void __fastcall sub_3030BD0(__int64 a1, int a2, __int64 a3)
{
  __int64 v6; // rsi
  __int64 *v7; // rax
  __int64 v8; // r10
  int v9; // r8d
  __int64 v10; // rdi
  int v11; // esi
  __int64 v12; // rcx
  int v13; // edx
  __int64 v14; // rax
  __int16 v15; // r15
  __int64 v16; // r13
  __int16 v17; // r11
  __int64 v18; // rax
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 *v24; // rax
  unsigned __int64 v25; // rcx
  __int64 v26; // rax
  __int64 *v27; // rax
  unsigned __int64 v28; // rcx
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 *v31; // rdi
  __int128 v32; // [rsp-20h] [rbp-120h]
  __int128 v33; // [rsp-10h] [rbp-110h]
  int v34; // [rsp+0h] [rbp-100h]
  int v35; // [rsp+0h] [rbp-100h]
  __int64 v36; // [rsp+0h] [rbp-100h]
  __int64 v37; // [rsp+8h] [rbp-F8h]
  __int64 v38; // [rsp+10h] [rbp-F0h] BYREF
  int v39; // [rsp+18h] [rbp-E8h]
  __int64 v40; // [rsp+20h] [rbp-E0h] BYREF
  int v41; // [rsp+28h] [rbp-D8h]
  __int64 v42; // [rsp+30h] [rbp-D0h]
  int v43; // [rsp+38h] [rbp-C8h]
  __int64 *v44; // [rsp+40h] [rbp-C0h]
  __int64 v45; // [rsp+48h] [rbp-B8h]
  __int64 v46; // [rsp+50h] [rbp-B0h] BYREF
  int v47; // [rsp+58h] [rbp-A8h]
  __int64 v48; // [rsp+60h] [rbp-A0h]
  int v49; // [rsp+68h] [rbp-98h]
  __int64 v50; // [rsp+70h] [rbp-90h]
  int v51; // [rsp+78h] [rbp-88h]
  __int16 *v52; // [rsp+80h] [rbp-80h]
  __int64 v53; // [rsp+88h] [rbp-78h]
  __int16 v54; // [rsp+90h] [rbp-70h] BYREF
  __int64 v55; // [rsp+98h] [rbp-68h]
  __int16 v56; // [rsp+A0h] [rbp-60h]
  __int64 v57; // [rsp+A8h] [rbp-58h]
  __int16 v58; // [rsp+B0h] [rbp-50h]
  __int64 v59; // [rsp+B8h] [rbp-48h]
  __int16 v60; // [rsp+C0h] [rbp-40h]
  __int64 v61; // [rsp+C8h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 80);
  v38 = v6;
  if ( v6 )
  {
    v34 = a2;
    sub_B96E90((__int64)&v38, v6, 1);
    a2 = v34;
  }
  v39 = *(_DWORD *)(a1 + 72);
  v7 = *(__int64 **)(a1 + 40);
  v8 = *v7;
  v9 = *((_DWORD *)v7 + 2);
  v52 = &v54;
  v10 = v7[5];
  v11 = *((_DWORD *)v7 + 12);
  v12 = v7[10];
  v13 = *((_DWORD *)v7 + 22);
  v14 = *(_QWORD *)(a1 + 48);
  v15 = *(_WORD *)(v14 + 16);
  v16 = *(_QWORD *)(v14 + 24);
  v46 = v8;
  v17 = *(_WORD *)(v14 + 32);
  v18 = *(_QWORD *)(v14 + 40);
  v51 = v13;
  v58 = v15;
  v61 = v18;
  v53 = 0x400000004LL;
  v54 = 8;
  v60 = v17;
  v59 = v16;
  v47 = v9;
  v48 = v10;
  v49 = v11;
  v50 = v12;
  v45 = 0x300000003LL;
  v55 = 0;
  v56 = 8;
  v57 = 0;
  v44 = &v46;
  *((_QWORD *)&v33 + 1) = 3;
  *(_QWORD *)&v33 = &v46;
  v35 = a2;
  v19 = sub_3411BE0(a2, 50, (unsigned int)&v38, (unsigned int)&v54, 4, a2, v33);
  *((_QWORD *)&v32 + 1) = 2;
  v40 = v19;
  v42 = v19;
  *(_QWORD *)&v32 = &v40;
  v41 = 0;
  v43 = 1;
  v21 = sub_33FC220(v35, 54, (unsigned int)&v38, 9, 0, v35, v32);
  v22 = *(unsigned int *)(a3 + 8);
  v23 = v20;
  if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v36 = v21;
    v37 = v20;
    sub_C8D5F0(a3, (const void *)(a3 + 16), v22 + 1, 0x10u, v21, v20);
    v22 = *(unsigned int *)(a3 + 8);
    v21 = v36;
    v23 = v37;
  }
  v24 = (__int64 *)(*(_QWORD *)a3 + 16 * v22);
  *v24 = v21;
  v24[1] = v23;
  v25 = *(unsigned int *)(a3 + 12);
  v26 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v26;
  if ( v26 + 1 > v25 )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v26 + 1, 0x10u, v21, v23);
    v26 = *(unsigned int *)(a3 + 8);
  }
  v27 = (__int64 *)(*(_QWORD *)a3 + 16 * v26);
  *v27 = v19;
  v27[1] = 2;
  v28 = *(unsigned int *)(a3 + 12);
  v29 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v29;
  if ( v29 + 1 > v28 )
  {
    sub_C8D5F0(a3, (const void *)(a3 + 16), v29 + 1, 0x10u, v21, v23);
    v29 = *(unsigned int *)(a3 + 8);
  }
  v30 = (__int64 *)(*(_QWORD *)a3 + 16 * v29);
  *v30 = v19;
  v31 = v44;
  v30[1] = 3;
  ++*(_DWORD *)(a3 + 8);
  if ( v31 != &v46 )
    _libc_free((unsigned __int64)v31);
  if ( v52 != &v54 )
    _libc_free((unsigned __int64)v52);
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
}

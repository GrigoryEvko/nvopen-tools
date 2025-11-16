// Function: sub_2775150
// Address: 0x2775150
//
unsigned __int64 __fastcall sub_2775150(__int64 a1, char *a2)
{
  char v4; // al
  __int64 v5; // rax
  __int64 v6; // r9
  unsigned int v7; // eax
  unsigned __int64 v8; // r12
  unsigned int v10; // esi
  int v11; // eax
  __int64 v12; // r13
  int v13; // eax
  __int64 v14; // r8
  char v15; // dl
  _QWORD *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // r15
  __int64 v20; // rax
  _QWORD *v21; // rdi
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r12
  unsigned __int64 v26; // rcx
  char *v27; // r15
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  int v30; // eax
  __int64 v31; // r12
  char v32; // al
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdi
  char *v36; // r15
  __int64 v37; // [rsp+18h] [rbp-B8h] BYREF
  char v38[8]; // [rsp+20h] [rbp-B0h] BYREF
  unsigned __int64 v39[2]; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v40; // [rsp+38h] [rbp-98h]
  unsigned __int64 v41[2]; // [rsp+40h] [rbp-90h] BYREF
  __int64 v42; // [rsp+50h] [rbp-80h]
  int v43; // [rsp+58h] [rbp-78h]
  __int64 v44; // [rsp+60h] [rbp-70h] BYREF
  unsigned __int64 v45; // [rsp+68h] [rbp-68h] BYREF
  __int64 v46; // [rsp+70h] [rbp-60h]
  __int64 v47; // [rsp+78h] [rbp-58h]
  unsigned __int64 v48; // [rsp+80h] [rbp-50h] BYREF
  __int64 v49; // [rsp+88h] [rbp-48h]
  __int64 v50; // [rsp+90h] [rbp-40h]
  __int64 v51; // [rsp+98h] [rbp-38h]

  v4 = *a2;
  v45 = 0;
  v46 = 0;
  LOBYTE(v44) = v4;
  v47 = *((_QWORD *)a2 + 3);
  if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
    sub_BD6050(&v45, *((_QWORD *)a2 + 1) & 0xFFFFFFFFFFFFFFF8LL);
  v5 = *((_QWORD *)a2 + 6);
  v48 = 0;
  v49 = 0;
  v50 = v5;
  if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
    sub_BD6050(&v48, *((_QWORD *)a2 + 4) & 0xFFFFFFFFFFFFFFF8LL);
  LODWORD(v51) = 0;
  v39[0] = 0;
  v38[0] = v44;
  v39[1] = 0;
  v40 = v47;
  if ( v47 != -4096 && v47 != 0 && v47 != -8192 )
    sub_BD6050(v39, v45 & 0xFFFFFFFFFFFFFFF8LL);
  v41[0] = 0;
  v41[1] = 0;
  v42 = v50;
  if ( v50 == -4096 || v50 == 0 || v50 == -8192 )
  {
    v43 = v51;
  }
  else
  {
    sub_BD6050(v41, v48 & 0xFFFFFFFFFFFFFFF8LL);
    v43 = v51;
    if ( v50 != 0 && v50 != -4096 && v50 != -8192 )
      sub_BD60C0(&v48);
  }
  if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
    sub_BD60C0(&v45);
  if ( (unsigned __int8)sub_27740B0(a1, v38, &v37) )
  {
    v7 = *(_DWORD *)(v37 + 56);
    goto LABEL_21;
  }
  v10 = *(_DWORD *)(a1 + 24);
  v11 = *(_DWORD *)(a1 + 16);
  v12 = v37;
  ++*(_QWORD *)a1;
  v13 = v11 + 1;
  v14 = 2 * v10;
  v44 = v12;
  if ( 4 * v13 >= 3 * v10 )
  {
    sub_2774A40(a1, v14);
LABEL_72:
    sub_27740B0(a1, v38, &v44);
    v12 = v44;
    v13 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_30;
  }
  if ( v10 - *(_DWORD *)(a1 + 20) - v13 <= v10 >> 3 )
  {
    sub_2774A40(a1, v10);
    goto LABEL_72;
  }
LABEL_30:
  *(_DWORD *)(a1 + 16) = v13;
  if ( !*(_BYTE *)v12 && !*(_QWORD *)(v12 + 24) && !*(_QWORD *)(v12 + 48) )
  {
    v16 = (_QWORD *)(v12 + 8);
    *(_BYTE *)v12 = v38[0];
    v18 = v40;
    if ( !v40 )
      goto LABEL_38;
LABEL_35:
    *(_QWORD *)(v12 + 24) = v18;
    if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
      sub_BD73F0((__int64)v16);
    goto LABEL_38;
  }
  v15 = v38[0];
  --*(_DWORD *)(a1 + 20);
  v16 = (_QWORD *)(v12 + 8);
  v17 = *(_QWORD *)(v12 + 24);
  *(_BYTE *)v12 = v15;
  v18 = v40;
  if ( v40 != v17 )
  {
    if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
    {
      sub_BD60C0(v16);
      v16 = (_QWORD *)(v12 + 8);
    }
    goto LABEL_35;
  }
LABEL_38:
  v19 = v42;
  v20 = *(_QWORD *)(v12 + 48);
  v21 = (_QWORD *)(v12 + 32);
  if ( v42 != v20 )
  {
    if ( v20 != -4096 && v20 != 0 && v20 != -8192 )
    {
      sub_BD60C0(v21);
      v21 = (_QWORD *)(v12 + 32);
    }
    *(_QWORD *)(v12 + 48) = v19;
    if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
      sub_BD73F0((__int64)v21);
  }
  *(_DWORD *)(v12 + 56) = v43;
  v22 = *a2;
  v45 = 0;
  LOBYTE(v44) = v22;
  v23 = *((_QWORD *)a2 + 3);
  v46 = 0;
  v47 = v23;
  if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
    sub_BD6050(&v45, *((_QWORD *)a2 + 1) & 0xFFFFFFFFFFFFFFF8LL);
  v24 = *((_QWORD *)a2 + 6);
  v48 = 0;
  v49 = 0;
  v50 = v24;
  if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
    sub_BD6050(&v48, *((_QWORD *)a2 + 4) & 0xFFFFFFFFFFFFFFF8LL);
  v25 = *(unsigned int *)(a1 + 40);
  v26 = *(unsigned int *)(a1 + 44);
  v51 = 0;
  v27 = (char *)&v44;
  v28 = *(_QWORD *)(a1 + 32);
  v29 = v25 + 1;
  v30 = v25;
  if ( v25 + 1 > v26 )
  {
    v35 = a1 + 32;
    if ( v28 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v28 + (v25 << 6) )
    {
      sub_2774890(v35, v29, v28, v26, v14, v6);
      v25 = *(unsigned int *)(a1 + 40);
      v28 = *(_QWORD *)(a1 + 32);
      v27 = (char *)&v44;
      v30 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      v36 = (char *)&v44 - v28;
      sub_2774890(v35, v29, v28, v26, v14, v6);
      v28 = *(_QWORD *)(a1 + 32);
      v25 = *(unsigned int *)(a1 + 40);
      v27 = &v36[v28];
      v30 = *(_DWORD *)(a1 + 40);
    }
  }
  v31 = v28 + (v25 << 6);
  if ( v31 )
  {
    v32 = *v27;
    *(_QWORD *)(v31 + 8) = 0;
    *(_QWORD *)(v31 + 16) = 0;
    *(_BYTE *)v31 = v32;
    v33 = *((_QWORD *)v27 + 3);
    *(_QWORD *)(v31 + 24) = v33;
    if ( v33 != -4096 && v33 != 0 && v33 != -8192 )
      sub_BD6050((unsigned __int64 *)(v31 + 8), *((_QWORD *)v27 + 1) & 0xFFFFFFFFFFFFFFF8LL);
    *(_QWORD *)(v31 + 32) = 0;
    v34 = *((_QWORD *)v27 + 6);
    *(_QWORD *)(v31 + 40) = 0;
    *(_QWORD *)(v31 + 48) = v34;
    if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
      sub_BD6050((unsigned __int64 *)(v31 + 32), *((_QWORD *)v27 + 4) & 0xFFFFFFFFFFFFFFF8LL);
    *(_QWORD *)(v31 + 56) = *((_QWORD *)v27 + 7);
    v30 = *(_DWORD *)(a1 + 40);
  }
  *(_DWORD *)(a1 + 40) = v30 + 1;
  if ( v50 != 0 && v50 != -4096 && v50 != -8192 )
    sub_BD60C0(&v48);
  if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
    sub_BD60C0(&v45);
  v7 = *(_DWORD *)(a1 + 40) - 1;
  *(_DWORD *)(v12 + 56) = v7;
LABEL_21:
  v8 = *(_QWORD *)(a1 + 32) + ((unsigned __int64)v7 << 6) + 56;
  if ( v42 != 0 && v42 != -4096 && v42 != -8192 )
    sub_BD60C0(v41);
  if ( v40 != 0 && v40 != -4096 && v40 != -8192 )
    sub_BD60C0(v39);
  return v8;
}

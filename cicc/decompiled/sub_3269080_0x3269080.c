// Function: sub_3269080
// Address: 0x3269080
//
__int64 __fastcall sub_3269080(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 v5; // r14
  __int16 *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rcx
  unsigned __int16 v9; // bx
  unsigned int v10; // r12d
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rdi
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r10
  __int64 v19; // rax
  int v20; // edx
  __int64 v21; // r8
  __int64 v22; // r10
  __int64 v23; // rdx
  __int64 v24; // rdi
  int v25; // eax
  int v26; // edx
  int v27; // r9d
  __int64 v28; // r13
  __int64 v29; // rsi
  __int64 v31; // rax
  char v32; // al
  __int128 v33; // [rsp-10h] [rbp-E0h]
  __int64 v34; // [rsp+0h] [rbp-D0h]
  __int64 v35; // [rsp+8h] [rbp-C8h]
  __int64 v36; // [rsp+10h] [rbp-C0h]
  int v37; // [rsp+1Ch] [rbp-B4h]
  __int64 v38; // [rsp+20h] [rbp-B0h]
  unsigned int v39; // [rsp+28h] [rbp-A8h]
  __int64 v40; // [rsp+28h] [rbp-A8h]
  __int64 v41; // [rsp+30h] [rbp-A0h]
  unsigned int v42; // [rsp+38h] [rbp-98h]
  __int64 v43; // [rsp+40h] [rbp-90h] BYREF
  int v44; // [rsp+48h] [rbp-88h]
  __int64 v45; // [rsp+50h] [rbp-80h] BYREF
  int v46; // [rsp+58h] [rbp-78h]
  __int64 v47; // [rsp+60h] [rbp-70h]
  __int64 v48; // [rsp+70h] [rbp-60h] BYREF
  int v49; // [rsp+78h] [rbp-58h]
  __int64 v50; // [rsp+80h] [rbp-50h]
  unsigned int v51; // [rsp+88h] [rbp-48h]
  __int64 v52; // [rsp+90h] [rbp-40h]
  int v53; // [rsp+98h] [rbp-38h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = v4[10];
  v36 = *v4;
  v37 = *((_DWORD *)v4 + 2);
  v41 = v4[5];
  v39 = *((_DWORD *)v4 + 12);
  v42 = *((_DWORD *)v4 + 22);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *((_QWORD *)v6 + 1);
  v9 = *v6;
  v10 = (unsigned __int16)v6[8];
  v43 = v7;
  v38 = v8;
  v35 = *((_QWORD *)v6 + 3);
  if ( v7 )
    sub_B96E90((__int64)&v43, v7, 1);
  v11 = *(_DWORD *)(a2 + 28);
  v44 = *(_DWORD *)(a2 + 72);
  v12 = *a1;
  v46 = v11;
  v13 = *(_QWORD *)(v12 + 1024);
  v45 = v12;
  v47 = v13;
  *(_QWORD *)(v12 + 1024) = &v45;
  v14 = *((unsigned __int8 *)a1 + 33);
  v15 = a1[1];
  if ( (_BYTE)v14 )
  {
    v16 = 1;
    if ( v9 != 1 )
    {
      if ( !v9 )
        goto LABEL_16;
      v16 = v9;
      if ( !*(_QWORD *)(v15 + 8LL * v9 + 112) )
        goto LABEL_21;
    }
    v17 = *(_BYTE *)(v15 + 500 * v16 + 6516);
    if ( v17 )
    {
      if ( v17 != 4 )
        goto LABEL_21;
    }
  }
  v18 = *a1;
  LODWORD(v48) = 2;
  v34 = v18;
  v19 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, __int64, __int64, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v15 + 2264LL))(
          v15,
          v5,
          v42,
          v18,
          v14,
          *((unsigned __int8 *)a1 + 35),
          &v48,
          0);
  if ( v19 )
  {
    if ( (int)v48 <= 0 )
    {
      v24 = *a1;
      v48 = v36;
      v49 = v37;
      v50 = v41;
      v51 = v39;
      goto LABEL_14;
    }
    if ( !*(_QWORD *)(v19 + 56) )
      sub_33ECEA0(v34, v19);
  }
  v21 = *((unsigned __int8 *)a1 + 33);
  v15 = a1[1];
  if ( (_BYTE)v21 )
  {
LABEL_21:
    v31 = 1;
    if ( v9 != 1 )
    {
      if ( !v9 )
        goto LABEL_16;
      v31 = v9;
      if ( !*(_QWORD *)(v15 + 8LL * v9 + 112) )
        goto LABEL_16;
    }
    v32 = *(_BYTE *)(v15 + 500 * v31 + 6516);
    if ( v32 )
    {
      if ( v32 != 4 )
        goto LABEL_16;
    }
    v21 = 1;
  }
  v22 = *a1;
  LODWORD(v48) = 2;
  v23 = v39;
  v40 = v22;
  v19 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, _QWORD, __int64 *, _QWORD))(*(_QWORD *)v15 + 2264LL))(
          v15,
          v41,
          v23,
          v22,
          v21,
          *((unsigned __int8 *)a1 + 35),
          &v48,
          0);
  if ( v19 )
  {
    if ( (int)v48 <= 0 )
    {
      v50 = v5;
      v24 = *a1;
      v48 = v36;
      v49 = v37;
      v51 = v42;
LABEL_14:
      v53 = v20;
      v52 = v19;
      v25 = sub_33E5110(v24, v9, v38, v10, v35);
      *((_QWORD *)&v33 + 1) = 3;
      *(_QWORD *)&v33 = &v48;
      v28 = sub_3411630(v24, 102, (unsigned int)&v43, v25, v26, v27, v33);
      goto LABEL_17;
    }
    if ( !*(_QWORD *)(v19 + 56) )
      sub_33ECEA0(v40, v19);
  }
LABEL_16:
  v28 = 0;
LABEL_17:
  v29 = v43;
  *(_QWORD *)(v45 + 1024) = v47;
  if ( v29 )
    sub_B91220((__int64)&v43, v29);
  return v28;
}

// Function: sub_2410910
// Address: 0x2410910
//
unsigned __int64 __fastcall sub_2410910(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int8 a6)
{
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned __int64 result; // rax
  __int64 v13; // r15
  char v14; // r14
  _QWORD *v15; // rax
  __int64 v16; // r9
  __int64 v17; // r12
  unsigned int *v18; // r15
  __int64 v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  unsigned int v22; // r15d
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rdi
  unsigned __int8 *v29; // r11
  __int64 (__fastcall *v30)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v31; // rax
  unsigned __int8 *v32; // r10
  __int64 v33; // rdi
  __int64 (__fastcall *v34)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v35; // rax
  unsigned int *v36; // r12
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // rax
  __int64 v40; // rbx
  char v41; // r15
  _QWORD *v42; // rax
  __int64 v43; // r9
  __int64 v44; // r12
  unsigned int *v45; // r14
  __int64 v46; // rbx
  __int64 v47; // rdx
  unsigned int v48; // esi
  __int64 v49; // r12
  unsigned int *v50; // rbx
  __int64 v51; // rdx
  unsigned int v52; // esi
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // [rsp-8h] [rbp-138h]
  unsigned __int64 v56; // [rsp+10h] [rbp-120h]
  unsigned int v57; // [rsp+1Ch] [rbp-114h]
  _BYTE *v58; // [rsp+20h] [rbp-110h]
  __int64 v59; // [rsp+28h] [rbp-108h]
  __int64 v60; // [rsp+28h] [rbp-108h]
  unsigned __int8 v63; // [rsp+40h] [rbp-F0h]
  unsigned __int64 v64; // [rsp+48h] [rbp-E8h]
  unsigned __int8 v65; // [rsp+48h] [rbp-E8h]
  unsigned int v66; // [rsp+48h] [rbp-E8h]
  __int64 v67; // [rsp+48h] [rbp-E8h]
  unsigned __int8 v68; // [rsp+48h] [rbp-E8h]
  unsigned int v71; // [rsp+68h] [rbp-C8h]
  unsigned __int8 *v72; // [rsp+68h] [rbp-C8h]
  unsigned __int8 *v73; // [rsp+68h] [rbp-C8h]
  __int64 v74; // [rsp+68h] [rbp-C8h]
  __int64 v75; // [rsp+68h] [rbp-C8h]
  unsigned __int8 *v76; // [rsp+68h] [rbp-C8h]
  unsigned __int8 *v77; // [rsp+68h] [rbp-C8h]
  char v78[32]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v79; // [rsp+90h] [rbp-A0h]
  char v80[32]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v81; // [rsp+C0h] [rbp-70h]
  unsigned __int64 v82; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v83; // [rsp+D8h] [rbp-58h]
  __int16 v84; // [rsp+F0h] [rbp-40h]

  v8 = sub_B2BEC0(a1[1]);
  v63 = sub_AE5020(v8, *(_QWORD *)(*a1 + 64));
  v9 = sub_9208B0(v8, *(_QWORD *)(*a1 + 64));
  v83 = v10;
  v82 = (unsigned __int64)(v9 + 7) >> 3;
  v11 = sub_CA1930(&v82);
  if ( v11 <= 4 || a6 < v63 )
    goto LABEL_3;
  v22 = v11;
  v23 = sub_B2BEC0(a1[1]);
  v24 = sub_9208B0(v23, *(_QWORD *)(*a1 + 64));
  v83 = v25;
  v82 = (unsigned __int64)(v24 + 7) >> 3;
  v59 = a3;
  if ( (unsigned int)sub_CA1930(&v82) == 4 )
    goto LABEL_32;
  v84 = 257;
  v26 = sub_921630((unsigned int **)a2, a3, *(_QWORD *)(*a1 + 64), 0, (__int64)&v82);
  v79 = 257;
  v81 = 257;
  v27 = sub_AD64C0(*(_QWORD *)(v26 + 8), 32, 0);
  v28 = *(_QWORD *)(a2 + 80);
  v29 = (unsigned __int8 *)v27;
  v30 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v28 + 32LL);
  if ( v30 != sub_9201A0 )
  {
    v77 = v29;
    v54 = v30(v28, 25u, (_BYTE *)v26, v29, 0, 0);
    v29 = v77;
    v32 = (unsigned __int8 *)v54;
    goto LABEL_20;
  }
  if ( *(_BYTE *)v26 <= 0x15u && *v29 <= 0x15u )
  {
    v72 = v29;
    if ( (unsigned __int8)sub_AC47B0(25) )
      v31 = sub_AD5570(25, v26, v72, 0, 0);
    else
      v31 = sub_AABE40(0x19u, (unsigned __int8 *)v26, v72);
    v29 = v72;
    v32 = (unsigned __int8 *)v31;
LABEL_20:
    if ( v32 )
      goto LABEL_21;
  }
  v84 = 257;
  v67 = sub_B504D0(25, v26, (__int64)v29, (__int64)&v82, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v67,
    v78,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v32 = (unsigned __int8 *)v67;
  v75 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v75 )
  {
    v68 = a6;
    v49 = (__int64)v32;
    v60 = v26;
    v50 = *(unsigned int **)a2;
    do
    {
      v51 = *((_QWORD *)v50 + 1);
      v52 = *v50;
      v50 += 4;
      sub_B99FD0(v49, v52, v51);
    }
    while ( (unsigned int *)v75 != v50 );
    v32 = (unsigned __int8 *)v49;
    v26 = v60;
    a6 = v68;
  }
LABEL_21:
  v33 = *(_QWORD *)(a2 + 80);
  v34 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v33 + 16LL);
  if ( v34 != sub_9202E0 )
  {
    v76 = v32;
    v53 = v34(v33, 29u, (_BYTE *)v26, v32);
    v32 = v76;
    v59 = v53;
    goto LABEL_27;
  }
  if ( *(_BYTE *)v26 <= 0x15u && *v32 <= 0x15u )
  {
    v73 = v32;
    if ( (unsigned __int8)sub_AC47B0(29) )
      v35 = sub_AD5570(29, v26, v73, 0, 0);
    else
      v35 = sub_AABE40(0x1Du, (unsigned __int8 *)v26, v73);
    v32 = v73;
    v59 = v35;
LABEL_27:
    if ( v59 )
      goto LABEL_32;
  }
  v84 = 257;
  v59 = sub_B504D0(29, v26, (__int64)v32, (__int64)&v82, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v59,
    v80,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v74 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v74 )
  {
    v65 = a6;
    v36 = *(unsigned int **)a2;
    do
    {
      v37 = *((_QWORD *)v36 + 1);
      v38 = *v36;
      v36 += 4;
      sub_B99FD0(v59, v38, v37);
    }
    while ( (unsigned int *)v74 != v36 );
    a6 = v65;
  }
LABEL_32:
  v84 = 257;
  v39 = sub_BCE3C0(*(__int64 **)(*a1 + 8), 0);
  v58 = sub_94BCF0((unsigned int **)a2, a4, v39, (__int64)&v82);
  v40 = (__int64)v58;
  v56 = a5 / v22;
  if ( a5 < v22 )
  {
LABEL_3:
    v71 = 0;
    result = 0;
    goto LABEL_4;
  }
  v71 = 0;
  v66 = 0;
  v57 = v22 >> 2;
  while ( 1 )
  {
    v41 = a6;
    v84 = 257;
    v42 = sub_BD2C40(80, unk_3F10A10);
    v44 = (__int64)v42;
    if ( v42 )
    {
      sub_B4D3C0((__int64)v42, v59, v40, 0, v41, v43, 0, 0);
      v43 = v55;
    }
    (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v44,
      &v82,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64),
      v43);
    v45 = *(unsigned int **)a2;
    v46 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v46 )
    {
      do
      {
        v47 = *((_QWORD *)v45 + 1);
        v48 = *v45;
        v45 += 4;
        sub_B99FD0(v44, v48, v47);
      }
      while ( (unsigned int *)v46 != v45 );
    }
    ++v66;
    v71 += v57;
    if ( v66 >= v56 )
      break;
    v40 = (__int64)v58;
    if ( v66 )
    {
      v84 = 257;
      v40 = sub_94B060((unsigned int **)a2, *(_QWORD *)(*a1 + 64), (__int64)v58, v66, (__int64)&v82);
    }
    a6 = v63;
  }
  result = v71;
  a6 = v63;
LABEL_4:
  v64 = (a5 + 3) >> 2;
  if ( v64 > result )
  {
    do
    {
      v13 = a4;
      if ( v71 )
      {
        v84 = 257;
        v13 = sub_94B060((unsigned int **)a2, *(_QWORD *)(*a1 + 24), a4, v71, (__int64)&v82);
      }
      v14 = a6;
      v84 = 257;
      v15 = sub_BD2C40(80, unk_3F10A10);
      v17 = (__int64)v15;
      if ( v15 )
        sub_B4D3C0((__int64)v15, a3, v13, 0, v14, v16, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v17,
        &v82,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      v18 = *(unsigned int **)a2;
      v19 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 != v19 )
      {
        do
        {
          v20 = *((_QWORD *)v18 + 1);
          v21 = *v18;
          v18 += 4;
          sub_B99FD0(v17, v21, v20);
        }
        while ( (unsigned int *)v19 != v18 );
      }
      ++v71;
      a6 = byte_4FE3AA8;
      result = v71;
    }
    while ( v71 < v64 );
  }
  return result;
}

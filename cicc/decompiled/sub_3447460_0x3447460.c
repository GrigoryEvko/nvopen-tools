// Function: sub_3447460
// Address: 0x3447460
//
__int64 __fastcall sub_3447460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6, __m128i a7)
{
  __int64 v10; // rsi
  unsigned int v11; // r15d
  bool v12; // al
  unsigned int v13; // r15d
  int v14; // r10d
  __int64 (*v15)(); // rax
  __int64 v16; // rax
  __int64 v17; // r12
  int v18; // eax
  int v20; // eax
  __int64 v21; // r13
  unsigned int v22; // r15d
  __int64 v23; // rax
  __int64 v24; // r13
  char v25; // al
  __int64 v26; // r8
  unsigned int v27; // r12d
  __int64 v28; // r11
  unsigned int v29; // eax
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rdx
  __int128 v32; // rax
  __int64 v33; // r8
  unsigned int v34; // r10d
  unsigned __int8 *v35; // rax
  int v36; // edx
  char v37; // al
  char v38; // al
  int v39; // eax
  __int64 v40; // [rsp+8h] [rbp-B8h]
  __int128 v41; // [rsp+10h] [rbp-B0h]
  __int128 v42; // [rsp+10h] [rbp-B0h]
  int v43; // [rsp+10h] [rbp-B0h]
  unsigned int v44; // [rsp+20h] [rbp-A0h]
  __int64 v45; // [rsp+20h] [rbp-A0h]
  __int64 v46; // [rsp+20h] [rbp-A0h]
  __int64 v47; // [rsp+20h] [rbp-A0h]
  int v48; // [rsp+28h] [rbp-98h]
  __int64 v49; // [rsp+28h] [rbp-98h]
  unsigned int v50; // [rsp+28h] [rbp-98h]
  unsigned int v51; // [rsp+28h] [rbp-98h]
  int v52; // [rsp+28h] [rbp-98h]
  int v53; // [rsp+28h] [rbp-98h]
  __int64 v56; // [rsp+60h] [rbp-60h] BYREF
  int v57; // [rsp+68h] [rbp-58h]
  unsigned __int64 v58; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v59; // [rsp+78h] [rbp-48h]
  unsigned __int64 v60; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v61; // [rsp+88h] [rbp-38h]

  v10 = *(_QWORD *)(a2 + 80);
  v56 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v56, v10, 1);
  v11 = *(_DWORD *)(a4 + 8);
  v57 = *(_DWORD *)(a2 + 72);
  if ( v11 <= 0x40 )
    v12 = *(_QWORD *)a4 == 0;
  else
    v12 = v11 == (unsigned int)sub_C444A0(a4);
  if ( v12 )
    goto LABEL_15;
  v13 = *(_DWORD *)(a5 + 8);
  LOBYTE(v13) = v13 <= 0x40 ? *(_QWORD *)a5 == 0 : v13 == (unsigned int)sub_C444A0(a5);
  if ( (_BYTE)v13 )
    goto LABEL_15;
  v14 = *(_DWORD *)(a2 + 24);
  v15 = *(__int64 (**)())(*(_QWORD *)a1 + 1992LL);
  if ( v15 != sub_302E0A0 )
  {
    v52 = *(_DWORD *)(a2 + 24);
    v37 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, __int64, __int64 *))v15)(a1, a2, a3, a4, a5, a6);
    v14 = v52;
    if ( v37 )
    {
      LOBYTE(v13) = a6[4] != 0;
      goto LABEL_16;
    }
  }
  if ( v14 == 187 )
    goto LABEL_13;
  if ( ((v14 - 186) & 0xFFFFFFFD) != 0 )
    goto LABEL_16;
  v16 = *(_QWORD *)(a2 + 56);
  if ( !v16 || *(_QWORD *)(v16 + 32) || (unsigned int)(*(_DWORD *)(*(_QWORD *)(v16 + 16) + 24LL) - 191) > 1 )
  {
LABEL_13:
    v17 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
    v18 = *(_DWORD *)(v17 + 24);
    if ( v18 != 35 && v18 != 11 )
      goto LABEL_15;
  }
  else
  {
    v17 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
    v20 = *(_DWORD *)(v17 + 24);
    if ( v20 != 11 && v20 != 35 )
      goto LABEL_15;
    v21 = *(_QWORD *)(v17 + 96);
    v22 = *(_DWORD *)(v21 + 32);
    if ( v22 > 0x40 )
    {
      v53 = v14;
      v39 = sub_9871A0(v21 + 24);
      v14 = v53;
      if ( v22 - v39 > 0x40 )
        goto LABEL_28;
      v23 = **(_QWORD **)(v21 + 24);
    }
    else
    {
      v23 = *(_QWORD *)(v21 + 24);
    }
    if ( v23 == 31 )
      goto LABEL_15;
  }
LABEL_28:
  if ( (*(_BYTE *)(v17 + 32) & 8) != 0 )
    goto LABEL_15;
  v24 = *(_QWORD *)(v17 + 96);
  if ( v14 == 188 )
  {
    if ( *(_DWORD *)(a4 + 8) <= 0x40u )
    {
      if ( (*(_QWORD *)a4 & ~*(_QWORD *)(v24 + 24)) == 0 )
        goto LABEL_15;
    }
    else
    {
      v38 = sub_C446F0((__int64 *)a4, (__int64 *)(v24 + 24));
      v14 = 188;
      if ( v38 )
        goto LABEL_15;
    }
  }
  if ( *(_DWORD *)(v24 + 32) > 0x40u )
  {
    v48 = v14;
    v25 = sub_C446F0((__int64 *)(v24 + 24), (__int64 *)a4);
    v14 = v48;
    if ( !v25 )
      goto LABEL_32;
LABEL_15:
    v13 = 0;
    goto LABEL_16;
  }
  if ( (*(_QWORD *)(v24 + 24) & ~*(_QWORD *)a4) == 0 )
    goto LABEL_15;
LABEL_32:
  v26 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3 + 8);
  v27 = *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3);
  v28 = *a6;
  v29 = *(_DWORD *)(a4 + 8);
  v59 = v29;
  if ( v29 <= 0x40 )
  {
    v30 = *(_QWORD *)a4;
LABEL_34:
    v58 = *(_QWORD *)(v24 + 24) & v30;
    v31 = v58;
    goto LABEL_35;
  }
  v40 = v26;
  v43 = v14;
  v47 = v28;
  sub_C43780((__int64)&v58, (const void **)a4);
  v29 = v59;
  v28 = v47;
  v14 = v43;
  v26 = v40;
  if ( v59 <= 0x40 )
  {
    v30 = v58;
    goto LABEL_34;
  }
  sub_C43B90(&v58, (__int64 *)(v24 + 24));
  v29 = v59;
  v31 = v58;
  v26 = v40;
  v14 = v43;
  v28 = v47;
LABEL_35:
  v60 = v31;
  v44 = v14;
  v49 = v26;
  v61 = v29;
  v59 = 0;
  *(_QWORD *)&v32 = sub_34007B0(v28, (__int64)&v60, (__int64)&v56, v27, v26, 0, a7, 0);
  v33 = v49;
  v34 = v44;
  if ( v61 > 0x40 && v60 )
  {
    v41 = v32;
    v45 = v49;
    v50 = v34;
    j_j___libc_free_0_0(v60);
    v32 = v41;
    v33 = v45;
    v34 = v50;
  }
  if ( v59 > 0x40 && v58 )
  {
    v42 = v32;
    v46 = v33;
    v51 = v34;
    j_j___libc_free_0_0(v58);
    v32 = v42;
    v33 = v46;
    v34 = v51;
  }
  v13 = 1;
  v35 = sub_3405C90(
          (_QWORD *)*a6,
          v34,
          (__int64)&v56,
          v27,
          v33,
          *(_DWORD *)(a2 + 28),
          a7,
          *(_OWORD *)*(_QWORD *)(a2 + 40),
          v32);
  a6[2] = a2;
  *((_DWORD *)a6 + 6) = a3;
  a6[4] = (__int64)v35;
  *((_DWORD *)a6 + 10) = v36;
LABEL_16:
  if ( v56 )
    sub_B91220((__int64)&v56, v56);
  return v13;
}

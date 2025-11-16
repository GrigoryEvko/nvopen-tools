// Function: sub_32730B0
// Address: 0x32730b0
//
__int64 __fastcall sub_32730B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int16 *v6; // rdx
  int v7; // eax
  __int64 v8; // rdx
  unsigned __int16 v9; // r13
  unsigned __int16 v10; // r15
  __int64 *v11; // rax
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r10
  __int64 v16; // r11
  __int64 v17; // rcx
  __int64 (*v18)(); // rax
  __int64 result; // rax
  char v20; // al
  __int64 (*v21)(); // rax
  __int64 v22; // rsi
  unsigned __int16 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // r13d
  int v27; // eax
  int v28; // esi
  __int128 v29; // rax
  int v30; // r9d
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r9
  __int64 v35; // r13
  __int64 v36; // rdx
  __int64 v37; // r14
  unsigned int v38; // r15d
  __int64 v39; // r8
  _BYTE *v40; // rcx
  __int64 v41; // rdx
  __int64 *v42; // rax
  unsigned __int16 v43; // ax
  int v44; // edx
  char v45; // al
  __int64 v46; // rdi
  __int64 v47; // rbx
  int v48; // edx
  __int64 v49; // rax
  _BYTE *v50; // rdx
  __int64 v51; // [rsp-10h] [rbp-150h]
  __int128 v52; // [rsp-10h] [rbp-150h]
  __int128 v53; // [rsp-10h] [rbp-150h]
  __int64 v54; // [rsp+0h] [rbp-140h]
  __int64 v55; // [rsp+0h] [rbp-140h]
  __int64 v56; // [rsp+8h] [rbp-138h]
  __int64 v57; // [rsp+8h] [rbp-138h]
  __int16 v58; // [rsp+1Ah] [rbp-126h]
  int v59; // [rsp+20h] [rbp-120h]
  unsigned int v61; // [rsp+34h] [rbp-10Ch]
  unsigned int v62; // [rsp+38h] [rbp-108h]
  __int64 v63; // [rsp+40h] [rbp-100h]
  __int128 v64; // [rsp+40h] [rbp-100h]
  __int64 v65; // [rsp+40h] [rbp-100h]
  __int64 v66; // [rsp+40h] [rbp-100h]
  __int64 v67; // [rsp+48h] [rbp-F8h]
  unsigned int v68; // [rsp+5Ch] [rbp-E4h] BYREF
  __int64 v69; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v70; // [rsp+68h] [rbp-D8h]
  __int16 v71; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v72; // [rsp+78h] [rbp-C8h]
  _BYTE *v73; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v74; // [rsp+88h] [rbp-B8h]
  _BYTE v75[176]; // [rsp+90h] [rbp-B0h] BYREF

  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOWORD(v69) = v7;
  v70 = v8;
  if ( (_WORD)v7 )
  {
    v59 = 0;
    v9 = word_4456580[v7 - 1];
  }
  else
  {
    v43 = sub_3009970((__int64)&v69, a2, v8, a4, a5);
    v59 = v44;
    v9 = v43;
  }
  v10 = v9;
  v62 = *(_DWORD *)(a2 + 24);
  v11 = *(__int64 **)(a2 + 40);
  v12 = *(_QWORD *)(*a1 + 16LL);
  v63 = *v11;
  v61 = *((_DWORD *)v11 + 2);
  v15 = sub_33F2320(*a1, *v11, v11[1], &v68);
  v16 = v13;
  if ( !v15 )
    return 0;
  v17 = v63;
  if ( *(_DWORD *)(v63 + 24) != 168 )
  {
    v18 = *(__int64 (**)())(*(_QWORD *)v12 + 1696LL);
    if ( v18 == sub_2FE35D0 )
      return 0;
    v54 = v15;
    v56 = v13;
    v20 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, _QWORD))v18)(v12, (unsigned int)v69, v70, v68);
    v15 = v54;
    v16 = v56;
    if ( !v20 )
      return 0;
  }
  if ( v9 != 1 && (!v9 || !*(_QWORD *)(v12 + 8LL * v9 + 112)) )
    return 0;
  if ( v62 <= 0x1F3 )
  {
    v14 = v12 + 500LL * v9;
    if ( (*(_BYTE *)(v62 + v14 + 6414) & 0xFB) != 0 )
      return 0;
  }
  v21 = *(__int64 (**)())(*(_QWORD *)v12 + 472LL);
  if ( v21 != sub_2FE30B0 )
  {
    v55 = v15;
    v57 = v16;
    v45 = ((__int64 (__fastcall *)(__int64, __int64))v21)(v12, a2);
    v15 = v55;
    v16 = v57;
    if ( !v45 )
      return 0;
  }
  v22 = v63;
  v23 = (unsigned __int16 *)(*(_QWORD *)(v63 + 48) + 16LL * v61);
  v24 = *v23;
  v25 = *((_QWORD *)v23 + 1);
  v71 = v24;
  v72 = v25;
  if ( (_WORD)v24 )
  {
    v26 = 0;
    LOWORD(v27) = word_4456580[(unsigned __int16)v24 - 1];
  }
  else
  {
    v66 = v15;
    v67 = v16;
    v27 = sub_3009970((__int64)&v71, v22, v24, v17, v14);
    v15 = v66;
    v16 = v67;
    v58 = HIWORD(v27);
    v26 = v48;
  }
  HIWORD(v28) = v58;
  *(_QWORD *)&v64 = v15;
  *((_QWORD *)&v64 + 1) = v16;
  LOWORD(v28) = v27;
  *(_QWORD *)&v29 = sub_3400EE0(*a1, (int)v68, a3, 0, v14);
  v31 = sub_3406EB0(*a1, 158, a3, v28, v26, v30, v64, v29);
  v33 = sub_33FA050(*a1, v62, a3, v10, v59, *(_DWORD *)(a2 + 28), v31, v32);
  v34 = v51;
  v35 = v33;
  v37 = v36;
  if ( (_WORD)v69 )
  {
    if ( (unsigned __int16)(v69 - 176) > 0x34u )
    {
      v38 = word_4456340[(unsigned __int16)v69 - 1];
      goto LABEL_20;
    }
  }
  else if ( !sub_3007100((__int64)&v69) )
  {
    v38 = sub_3007130((__int64)&v69, v62);
LABEL_20:
    v39 = v38;
    v73 = v75;
    v74 = 0x800000000LL;
    if ( v38 > 8 )
    {
      sub_C8D5F0((__int64)&v73, v75, v38, 0x10u, v38, v34);
      v39 = v38;
      v49 = (__int64)v73;
      v50 = &v73[16 * v38];
      do
      {
        if ( v49 )
        {
          *(_QWORD *)v49 = v35;
          *(_DWORD *)(v49 + 8) = v37;
        }
        v49 += 16;
      }
      while ( v50 != (_BYTE *)v49 );
      LODWORD(v74) = v38;
      v40 = v73;
    }
    else
    {
      v40 = v75;
      if ( v38 )
      {
        v41 = v38;
        v42 = (__int64 *)v75;
        do
        {
          *v42 = v35;
          v42 += 2;
          *((_DWORD *)v42 - 2) = v37;
          --v41;
        }
        while ( v41 );
        v40 = v73;
      }
      LODWORD(v74) = v38;
    }
    *((_QWORD *)&v52 + 1) = v39;
    *(_QWORD *)&v52 = v40;
    result = sub_33FC220(*a1, 156, a3, v69, v70, v34, v52);
    if ( v73 != v75 )
    {
      v65 = result;
      _libc_free((unsigned __int64)v73);
      return v65;
    }
    return result;
  }
  v46 = *a1;
  if ( *(_DWORD *)(v35 + 24) == 51 )
  {
    v73 = 0;
    LODWORD(v74) = 0;
    v47 = sub_33F17F0(v46, 51, &v73, v69, v70);
    if ( v73 )
      sub_B91220((__int64)&v73, (__int64)v73);
  }
  else
  {
    *((_QWORD *)&v53 + 1) = v37;
    *(_QWORD *)&v53 = v35;
    return sub_33FAF80(v46, 168, a3, v69, v70, v34, v53);
  }
  return v47;
}

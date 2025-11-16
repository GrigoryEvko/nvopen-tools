// Function: sub_345A180
// Address: 0x345a180
//
unsigned __int8 *__fastcall sub_345A180(__m128i a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rsi
  unsigned int *v7; // rax
  __int64 v8; // rcx
  unsigned __int16 *v9; // rax
  int v10; // r13d
  __int64 v11; // r14
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r13
  unsigned __int16 *v17; // rcx
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  int v20; // eax
  unsigned int v21; // r14d
  int v22; // esi
  unsigned __int16 v23; // ax
  __int64 v24; // rcx
  bool v25; // zf
  __int128 v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned __int8 *v29; // rax
  unsigned int v30; // ecx
  __int64 v31; // r13
  unsigned __int8 *v32; // r10
  __int64 v33; // rdx
  __int64 v34; // r11
  __int64 v35; // rdx
  __int64 v36; // r8
  __int16 v37; // ax
  __int64 v38; // rdx
  __int64 v39; // r9
  unsigned int v40; // esi
  int v41; // r9d
  unsigned __int8 *v42; // rax
  unsigned int v43; // edx
  unsigned __int8 *v44; // r12
  unsigned int v46; // eax
  bool v47; // al
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // rdx
  __int64 (__fastcall *v53)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v54; // rsi
  __int64 v55; // rcx
  __int64 v56; // r8
  int v57; // eax
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int128 v60; // [rsp-20h] [rbp-130h]
  __int64 v61; // [rsp+8h] [rbp-108h]
  unsigned int v62; // [rsp+8h] [rbp-108h]
  int v63; // [rsp+10h] [rbp-100h]
  __int128 v64; // [rsp+10h] [rbp-100h]
  __int64 v65; // [rsp+20h] [rbp-F0h]
  __int64 v66; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v67; // [rsp+30h] [rbp-E0h]
  __int64 v68; // [rsp+38h] [rbp-D8h]
  __int64 v69; // [rsp+40h] [rbp-D0h]
  unsigned __int16 v70; // [rsp+48h] [rbp-C8h]
  __int64 v71; // [rsp+58h] [rbp-B8h]
  __int64 v72; // [rsp+60h] [rbp-B0h] BYREF
  int v73; // [rsp+68h] [rbp-A8h]
  unsigned __int16 v74; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v75; // [rsp+78h] [rbp-98h]
  __int16 v76; // [rsp+80h] [rbp-90h] BYREF
  __int64 v77; // [rsp+88h] [rbp-88h]
  __int64 v78; // [rsp+90h] [rbp-80h] BYREF
  __int64 v79; // [rsp+98h] [rbp-78h]
  unsigned __int64 v80; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v81; // [rsp+A8h] [rbp-68h]
  unsigned __int64 v82; // [rsp+B0h] [rbp-60h]
  unsigned int v83; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v84; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v85; // [rsp+C8h] [rbp-48h]
  __int64 v86; // [rsp+D0h] [rbp-40h]
  unsigned int v87; // [rsp+D8h] [rbp-38h]

  v6 = *(_QWORD *)(a3 + 80);
  v72 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v72, v6, 1);
  v73 = *(_DWORD *)(a3 + 72);
  v7 = *(unsigned int **)(a3 + 40);
  v8 = *(_QWORD *)v7;
  v61 = v7[2];
  v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16 * v61);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v65 = v8;
  v74 = v10;
  v75 = v11;
  if ( (_WORD)v10 )
  {
    if ( (unsigned __int16)(v10 - 17) <= 0xD3u )
    {
      v11 = 0;
      LOWORD(v10) = word_4456580[v10 - 1];
    }
  }
  else if ( sub_30070B0((__int64)&v74) )
  {
    LOWORD(v10) = sub_3009970((__int64)&v74, v6, v49, v50, v51);
    v11 = v52;
  }
  v77 = v11;
  v76 = v10;
  sub_AADB10((__int64)&v80, 1u, 1);
  v14 = v74;
  if ( !v74 )
  {
    if ( !sub_3007100((__int64)&v74) )
    {
      v16 = *(_QWORD *)(a4 + 16);
      goto LABEL_12;
    }
LABEL_37:
    sub_988CD0((__int64)&v84, **(_QWORD **)(a4 + 40), 0x40u);
    if ( v81 > 0x40 && v80 )
      j_j___libc_free_0_0(v80);
    v80 = v84;
    v46 = v85;
    LODWORD(v85) = 0;
    v81 = v46;
    if ( v83 > 0x40 && v82 )
    {
      j_j___libc_free_0_0(v82);
      v82 = v86;
      v83 = v87;
      if ( (unsigned int)v85 > 0x40 && v84 )
        j_j___libc_free_0_0(v84);
    }
    else
    {
      v82 = v86;
      v83 = v87;
    }
    v14 = v74;
    v16 = *(_QWORD *)(a4 + 16);
    if ( v74 )
    {
      v15 = (unsigned int)v74 - 176;
      goto LABEL_9;
    }
LABEL_12:
    v18 = sub_3007240((__int64)&v74);
    HIDWORD(v71) = HIDWORD(v18);
    v15 = HIDWORD(v18);
    goto LABEL_13;
  }
  v15 = (unsigned int)v74 - 176;
  if ( (unsigned __int16)(v74 - 176) <= 0x34u )
    goto LABEL_37;
  v16 = *(_QWORD *)(a4 + 16);
LABEL_9:
  v17 = word_4456340;
  LOBYTE(v15) = (unsigned __int16)v15 <= 0x34u;
  LODWORD(v18) = word_4456340[v14 - 1];
LABEL_13:
  BYTE4(v71) = v15;
  LODWORD(v71) = v18;
  v19 = sub_3007410((__int64)&v76, *(__int64 **)(a4 + 64), v15, (__int64)v17, v12, v13);
  v20 = sub_2FE69A0(v16, v19, v71, 1, (__int64)&v80);
  v70 = 2;
  if ( v20 != 1 )
  {
    v70 = 3;
    if ( v20 != 2 )
    {
      v70 = 4;
      if ( v20 != 4 )
      {
        v70 = 5;
        if ( v20 != 8 )
        {
          v70 = 6;
          if ( v20 != 16 )
          {
            v70 = 7;
            if ( v20 != 32 )
            {
              v70 = 8;
              if ( v20 != 64 )
                v70 = 9 * (v20 == 128);
            }
          }
        }
      }
    }
  }
  v21 = v70;
  if ( v74 )
  {
    v22 = word_4456340[v74 - 1];
    if ( (unsigned __int16)(v74 - 176) > 0x34u )
      v23 = sub_2D43050(v70, v22);
    else
      v23 = sub_2D43AD0(v70, v22);
    v24 = 0;
  }
  else
  {
    v23 = sub_3009490(&v74, v70, 0);
    v24 = v48;
  }
  LOWORD(v78) = v23;
  v25 = *(_BYTE *)(v16 + v23 + 524896) == 1;
  v79 = v24;
  v69 = 0;
  if ( v25 )
  {
    v53 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v16 + 592LL);
    if ( v53 == sub_2D56A50 )
    {
      v54 = v16;
      sub_2FE6CC0((__int64)&v84, v16, *(_QWORD *)(a4 + 64), v78, v79);
      LOWORD(v57) = v85;
      v58 = v86;
      LOWORD(v78) = v85;
      v79 = v86;
    }
    else
    {
      v54 = *(_QWORD *)(a4 + 64);
      v57 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v53)(v16, v54, (unsigned int)v78);
      LODWORD(v78) = v57;
      v79 = v58;
    }
    if ( (_WORD)v57 )
    {
      v69 = 0;
      v70 = word_4456580[(unsigned __int16)v57 - 1];
    }
    else
    {
      v63 = sub_3009970((__int64)&v78, v54, v58, v55, v56);
      v70 = v63;
      v69 = v59;
    }
    HIWORD(v21) = HIWORD(v63);
  }
  *(_QWORD *)&v26 = sub_3400BD0(a4, 0, (__int64)&v72, (unsigned int)v78, v79, 0, a1, 0);
  v64 = v26;
  v29 = sub_3402A00((_QWORD *)a4, (unsigned __int64 *)&v72, (unsigned int)v78, v79, a1, v27, v28);
  v30 = v78;
  v31 = v79;
  v32 = v29;
  v34 = v33;
  v35 = *(_QWORD *)(v65 + 48) + 16 * v61;
  v36 = v65;
  v37 = *(_WORD *)v35;
  v38 = *(_QWORD *)(v35 + 8);
  v39 = v61;
  LOWORD(v84) = v37;
  v85 = v38;
  if ( v37 )
  {
    v40 = ((unsigned __int16)(v37 - 17) < 0xD4u) + 205;
  }
  else
  {
    v62 = v78;
    v66 = v39;
    v67 = v32;
    v68 = v34;
    v47 = sub_30070B0((__int64)&v84);
    v30 = v62;
    v36 = v65;
    v39 = v66;
    v32 = v67;
    v34 = v68;
    v40 = 205 - (!v47 - 1);
  }
  *((_QWORD *)&v60 + 1) = v34;
  *(_QWORD *)&v60 = v32;
  sub_340EC60((_QWORD *)a4, v40, (__int64)&v72, v30, v31, 0, v36, v39, v60, v64);
  LOWORD(v21) = v70;
  v42 = sub_33FAF80(a4, 389, (__int64)&v72, v21, v69, v41, a1);
  v44 = sub_33FB310(
          a4,
          (__int64)v42,
          v43,
          (__int64)&v72,
          **(unsigned __int16 **)(a3 + 48),
          *(_QWORD *)(*(_QWORD *)(a3 + 48) + 8LL),
          a1);
  if ( v83 > 0x40 && v82 )
    j_j___libc_free_0_0(v82);
  if ( v81 > 0x40 && v80 )
    j_j___libc_free_0_0(v80);
  if ( v72 )
    sub_B91220((__int64)&v72, v72);
  return v44;
}

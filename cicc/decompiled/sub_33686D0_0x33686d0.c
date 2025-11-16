// Function: sub_33686D0
// Address: 0x33686d0
//
__int64 __fastcall sub_33686D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 result; // rax
  __int64 v12; // rcx
  unsigned __int16 v13; // r15
  _WORD *v14; // rcx
  int v15; // eax
  unsigned __int64 v16; // rax
  unsigned int v17; // r10d
  unsigned __int64 v18; // r9
  unsigned int v19; // r11d
  unsigned __int64 v20; // rax
  int v21; // esi
  int v22; // eax
  unsigned int v23; // r11d
  int v24; // r9d
  unsigned int v25; // r10d
  int v26; // r8d
  int v27; // ebx
  __int64 v28; // rax
  unsigned int v29; // edx
  __int128 v30; // rax
  __int128 v31; // rax
  int v32; // r9d
  __int64 v33; // rcx
  __int64 v34; // r8
  unsigned __int16 *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int16 v38; // ax
  __int16 v39; // ax
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  unsigned int v43; // r11d
  __int64 v44; // r13
  int v45; // edx
  int v46; // r12d
  __int64 v47; // r8
  unsigned int v48; // r10d
  __int64 v49; // rdx
  unsigned int v50; // r10d
  int v51; // ebx
  __int64 v52; // r12
  unsigned __int64 v53; // r9
  int v54; // ecx
  __int64 v55; // rsi
  _BYTE *v56; // rdx
  int v57; // edx
  __int128 v58; // [rsp-20h] [rbp-200h]
  __int128 v59; // [rsp-10h] [rbp-1F0h]
  __int128 v60; // [rsp-10h] [rbp-1F0h]
  __int64 v61; // [rsp-8h] [rbp-1E8h]
  __int16 v62; // [rsp+Ah] [rbp-1D6h]
  unsigned int v63; // [rsp+14h] [rbp-1CCh]
  unsigned int v64; // [rsp+18h] [rbp-1C8h]
  unsigned int v65; // [rsp+18h] [rbp-1C8h]
  char v66; // [rsp+18h] [rbp-1C8h]
  unsigned int v67; // [rsp+18h] [rbp-1C8h]
  __int64 v68; // [rsp+20h] [rbp-1C0h]
  unsigned __int8 v69; // [rsp+20h] [rbp-1C0h]
  char v70; // [rsp+20h] [rbp-1C0h]
  unsigned int v71; // [rsp+20h] [rbp-1C0h]
  unsigned __int8 v72; // [rsp+20h] [rbp-1C0h]
  __int64 v73; // [rsp+28h] [rbp-1B8h]
  __int16 v74; // [rsp+30h] [rbp-1B0h]
  unsigned int v75; // [rsp+30h] [rbp-1B0h]
  unsigned int v76; // [rsp+30h] [rbp-1B0h]
  __int128 v77; // [rsp+30h] [rbp-1B0h]
  unsigned int v78; // [rsp+30h] [rbp-1B0h]
  unsigned int v79; // [rsp+30h] [rbp-1B0h]
  unsigned __int16 v80; // [rsp+40h] [rbp-1A0h]
  __int128 v81; // [rsp+40h] [rbp-1A0h]
  unsigned int v82; // [rsp+40h] [rbp-1A0h]
  __int64 v84; // [rsp+58h] [rbp-188h]
  __int64 v85; // [rsp+60h] [rbp-180h] BYREF
  __int64 v86; // [rsp+68h] [rbp-178h]
  unsigned __int16 v87; // [rsp+80h] [rbp-160h] BYREF
  __int64 v88; // [rsp+88h] [rbp-158h]
  __int64 v89; // [rsp+90h] [rbp-150h] BYREF
  int v90; // [rsp+98h] [rbp-148h]
  _BYTE *v91; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v92; // [rsp+A8h] [rbp-138h]
  _BYTE v93[304]; // [rsp+B0h] [rbp-130h] BYREF

  v6 = (unsigned int)a3;
  v9 = a2;
  v85 = a5;
  v86 = a6;
  if ( (_WORD)a5 )
  {
    v10 = (unsigned int)(unsigned __int16)a5 - 17;
    if ( (unsigned __int16)(a5 - 17) > 0xD3u )
      return 0;
    v73 = 0;
    v12 = *(_QWORD *)(a2 + 48) + 16 * v6;
    v13 = *(_WORD *)v12;
    v88 = *(_QWORD *)(v12 + 8);
    v14 = word_4456580;
    v87 = v13;
    v80 = word_4456580[(unsigned __int16)a5 - 1];
    if ( v13 )
    {
LABEL_6:
      v68 = 0;
      v74 = word_4456580[v13 - 1];
      v15 = (unsigned __int16)v85;
      if ( !(_WORD)v85 )
        goto LABEL_7;
      goto LABEL_25;
    }
  }
  else
  {
    if ( !sub_30070B0((__int64)&v85) )
      return 0;
    v35 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16 * v6);
    v36 = *v35;
    v37 = *((_QWORD *)v35 + 1);
    v87 = v36;
    v88 = v37;
    v38 = sub_3009970((__int64)&v85, a2, v36, v33, v34);
    v13 = v87;
    v80 = v38;
    v73 = v10;
    if ( v87 )
      goto LABEL_6;
  }
  v39 = sub_3009970((__int64)&v87, a2, v10, (__int64)v14, a5);
  v13 = v87;
  v74 = v39;
  v15 = (unsigned __int16)v85;
  v68 = v40;
  if ( !(_WORD)v85 )
  {
LABEL_7:
    v16 = sub_3007240((__int64)&v85);
    v17 = v16;
    v18 = HIDWORD(v16);
    if ( v13 )
      goto LABEL_8;
LABEL_26:
    v63 = v17;
    v66 = v18;
    v41 = sub_3007240((__int64)&v87);
    LOBYTE(v18) = v66;
    v17 = v63;
    v19 = v41;
    v20 = HIDWORD(v41);
    if ( v66 )
      goto LABEL_9;
LABEL_27:
    if ( v17 <= v19 || (_BYTE)v20 )
      return 0;
    goto LABEL_11;
  }
LABEL_25:
  LOBYTE(v18) = (unsigned __int16)(v15 - 176) <= 0x34u;
  v17 = word_4456340[v15 - 1];
  if ( !v13 )
    goto LABEL_26;
LABEL_8:
  v19 = word_4456340[v13 - 1];
  LOBYTE(v20) = (unsigned __int16)(v13 - 176) <= 0x34u;
  if ( !(_BYTE)v18 )
    goto LABEL_27;
LABEL_9:
  if ( !(_BYTE)v20 || v17 <= v19 )
    return 0;
LABEL_11:
  if ( v80 == 11 && v74 == 10 )
  {
    if ( v13 )
    {
      v64 = v17;
      v69 = v18;
      v75 = v19;
      v21 = word_4456340[v13 - 1];
      if ( (unsigned __int16)(v13 - 176) > 0x34u )
      {
        LOWORD(v22) = sub_2D43050(11, v21);
        v25 = v64;
        v24 = v69;
        v23 = v75;
      }
      else
      {
        LOWORD(v22) = sub_2D43AD0(11, v21);
        v23 = v75;
        v24 = v69;
        v25 = v64;
      }
      v26 = 0;
    }
    else
    {
      v67 = v17;
      v72 = v18;
      v79 = v19;
      v22 = sub_3009490(&v87, 0xBu, 0);
      v25 = v67;
      v24 = v72;
      v62 = HIWORD(v22);
      v23 = v79;
      v26 = v57;
    }
    HIWORD(v27) = v62;
    *((_QWORD *)&v59 + 1) = a3;
    *(_QWORD *)&v59 = a2;
    LOWORD(v27) = v22;
    v65 = v25;
    v70 = v24;
    v76 = v23;
    v28 = sub_33FAF80(a1, 234, a4, v27, v26, v24, v59);
    v19 = v76;
    LOBYTE(v18) = v70;
    v9 = v28;
    v6 = v29;
    v17 = v65;
  }
  else if ( v80 != v74 || v73 != v68 && !v80 )
  {
    return 0;
  }
  if ( (_BYTE)v18 )
  {
    *(_QWORD *)&v30 = sub_3400EE0(a1, 0, a4, 0, a5);
    v81 = v30;
    v91 = 0;
    LODWORD(v92) = 0;
    *(_QWORD *)&v31 = sub_33F17F0(a1, 51, &v91, v85, v86);
    if ( v91 )
    {
      v77 = v31;
      sub_B91220((__int64)&v91, (__int64)v91);
      v31 = v77;
    }
    *((_QWORD *)&v58 + 1) = v6 | a3 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v58 = v9;
    return sub_340F900(a1, 160, a4, v85, v86, v32, v31, v58, v81);
  }
  else
  {
    v71 = v17;
    v92 = 0x1000000000LL;
    v78 = v19;
    v91 = v93;
    sub_3408690(a1, v9, v6, (unsigned int)&v91, 0, 0, 0, 0);
    v89 = 0;
    v90 = 0;
    v42 = sub_33F17F0(a1, 51, &v89, v80, v73);
    v43 = v78;
    v44 = v42;
    v46 = v45;
    v47 = v61;
    v48 = v71;
    if ( v89 )
    {
      sub_B91220((__int64)&v89, v89);
      v48 = v71;
      v43 = v78;
    }
    v49 = (unsigned int)v92;
    v50 = v48 - v43;
    v51 = v46;
    v52 = v50;
    v53 = v50 + (unsigned __int64)(unsigned int)v92;
    v54 = v92;
    if ( v53 > HIDWORD(v92) )
    {
      v82 = v50;
      sub_C8D5F0((__int64)&v91, v93, v50 + (unsigned __int64)(unsigned int)v92, 0x10u, v47, v53);
      v49 = (unsigned int)v92;
      v50 = v82;
      v54 = v92;
    }
    v55 = (__int64)v91;
    v56 = &v91[16 * v49];
    if ( v52 )
    {
      do
      {
        if ( v56 )
        {
          *(_QWORD *)v56 = v44;
          *((_DWORD *)v56 + 2) = v51;
        }
        v56 += 16;
        --v52;
      }
      while ( v52 );
      v55 = (__int64)v91;
      v54 = v92;
    }
    LODWORD(v92) = v54 + v50;
    *((_QWORD *)&v60 + 1) = v54 + v50;
    *(_QWORD *)&v60 = v55;
    result = sub_33FC220(a1, 156, a4, v85, v86, v53, v60);
    if ( v91 != v93 )
    {
      v84 = result;
      _libc_free((unsigned __int64)v91);
      return v84;
    }
  }
  return result;
}

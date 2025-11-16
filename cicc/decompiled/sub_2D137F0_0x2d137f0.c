// Function: sub_2D137F0
// Address: 0x2d137f0
//
void __fastcall sub_2D137F0(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 v4; // r9
  unsigned __int64 v5; // r10
  __int64 v6; // r11
  __int64 v7; // rdx
  __int64 v8; // rax
  _QWORD *v9; // r13
  _QWORD *v10; // rdi
  unsigned __int64 v11; // r14
  __int64 v12; // r8
  unsigned __int64 v13; // r12
  __int64 v14; // rcx
  __int64 v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // r14
  unsigned __int64 *v20; // r13
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  _QWORD *v25; // r13
  unsigned __int64 v26; // r14
  _QWORD *v27; // r10
  __int64 v28; // r8
  unsigned __int64 v29; // r9
  __int64 v30; // rcx
  __int64 v31; // r11
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rbx
  __int64 v36; // rsi
  unsigned __int64 *v37; // rax
  unsigned __int64 v38; // rbx
  unsigned __int64 v39; // rcx
  _QWORD *v40; // r11
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned __int64 *v44; // [rsp-108h] [rbp-108h]
  unsigned __int64 v45; // [rsp-100h] [rbp-100h]
  _QWORD *v46; // [rsp-100h] [rbp-100h]
  unsigned __int64 v47; // [rsp-100h] [rbp-100h]
  __int64 v48; // [rsp-F8h] [rbp-F8h]
  __int64 v49; // [rsp-F8h] [rbp-F8h]
  unsigned __int64 v50; // [rsp-F8h] [rbp-F8h]
  unsigned __int64 v51; // [rsp-F8h] [rbp-F8h]
  __int64 v52; // [rsp-F0h] [rbp-F0h]
  __int64 v53; // [rsp-F0h] [rbp-F0h]
  _QWORD *v54; // [rsp-F0h] [rbp-F0h]
  __int64 v55; // [rsp-F0h] [rbp-F0h]
  __int64 v56; // [rsp-F0h] [rbp-F0h]
  __int64 v57; // [rsp-D8h] [rbp-D8h]
  __int64 v58; // [rsp-D0h] [rbp-D0h]
  unsigned __int64 v59; // [rsp-C8h] [rbp-C8h]
  unsigned __int64 v60; // [rsp-C8h] [rbp-C8h]
  unsigned __int64 v61; // [rsp-C8h] [rbp-C8h]
  __int64 v62; // [rsp-C8h] [rbp-C8h]
  __int64 v63; // [rsp-C8h] [rbp-C8h]
  __int64 v64; // [rsp-C8h] [rbp-C8h]
  unsigned __int64 v65; // [rsp-C0h] [rbp-C0h]
  __int64 v66; // [rsp-C0h] [rbp-C0h]
  _QWORD v67[3]; // [rsp-B8h] [rbp-B8h] BYREF
  unsigned __int64 v68; // [rsp-A0h] [rbp-A0h]
  __int64 v69; // [rsp-98h] [rbp-98h] BYREF
  unsigned __int64 v70; // [rsp-90h] [rbp-90h]
  __int64 v71; // [rsp-88h] [rbp-88h]
  _QWORD *v72; // [rsp-80h] [rbp-80h]
  __int64 v73; // [rsp-78h] [rbp-78h] BYREF
  unsigned __int64 v74; // [rsp-70h] [rbp-70h]
  __int64 v75; // [rsp-68h] [rbp-68h]
  _QWORD *v76; // [rsp-60h] [rbp-60h]
  __int64 v77; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int64 v78; // [rsp-50h] [rbp-50h]
  __int64 v79; // [rsp-48h] [rbp-48h]
  _QWORD *v80; // [rsp-40h] [rbp-40h]

  if ( a2 == a1 )
    return;
  v4 = a1[5];
  v5 = a1[4];
  v6 = a1[2];
  v7 = (__int64)(a1[9] - v4) >> 3;
  v8 = (__int64)(a1[6] - a1[7]) >> 3;
  v9 = (_QWORD *)a2[9];
  v10 = (_QWORD *)a2[5];
  v11 = a2[7];
  v12 = a2[4];
  v13 = a2[3];
  v14 = a2[6];
  v15 = a2[2];
  v16 = ((v7 - 1) << 6) + v8 + ((__int64)(v5 - v6) >> 3);
  v59 = a1[3];
  if ( v16 < ((v12 - v15) >> 3) + ((v9 - v10 - 1) << 6) + ((__int64)(v14 - v11) >> 3) )
  {
    v23 = v16 + ((__int64)(v15 - v13) >> 3);
    if ( v23 < 0 )
    {
      v24 = ~((unsigned __int64)~v23 >> 6);
    }
    else
    {
      if ( v23 <= 63 )
      {
        v66 = v12;
        v25 = v10;
        v26 = v13;
        v58 = v15 + 8 * v16;
LABEL_10:
        v72 = v10;
        v70 = v13;
        v78 = v59;
        v69 = v15;
        v73 = v58;
        v77 = v6;
        v79 = v5;
        v80 = (_QWORD *)v4;
        v75 = v66;
        v71 = v12;
        v74 = v26;
        v76 = v25;
        sub_2769DA0(v67, (__int64)&v69, &v73, (__int64)&v77);
        v27 = (_QWORD *)a2[9];
        v28 = a2[6];
        v29 = a2[7];
        v30 = a1[6];
        v31 = a1[8];
        v57 = a2[8];
        v32 = ((v27 - v25 - 1) << 6) + ((__int64)(v28 - v29) >> 3) + ((v66 - v58) >> 3);
        v33 = a1[2];
        if ( v30 != v33 )
        {
          v34 = ((v31 - v30) >> 3) - 1;
          if ( v32 > v34 )
          {
            v47 = a2[9];
            v51 = a2[7];
            v56 = a2[6];
            v64 = ((((__int64)(v47 - (_QWORD)v25) >> 3) - 1) << 6) + ((__int64)(v56 - v29) >> 3) + ((v66 - v58) >> 3);
            sub_2769620(a1, v32 - v34);
            v30 = a1[6];
            v31 = a1[8];
            v27 = (_QWORD *)v47;
            v29 = v51;
            v28 = v56;
            v32 = v64;
          }
          v35 = a1[7];
          v45 = v35;
          v61 = a1[9];
          v36 = v32 + ((__int64)(v30 - v35) >> 3);
          if ( v36 < 0 )
          {
            v52 = ~((unsigned __int64)~v36 >> 6);
          }
          else
          {
            if ( v36 <= 63 )
            {
              v48 = v31;
              v53 = v30 + 8 * v32;
              v37 = (unsigned __int64 *)a1[9];
              goto LABEL_17;
            }
            v52 = v36 >> 6;
          }
          v37 = (unsigned __int64 *)(v61 + 8 * v52);
          v35 = *v37;
          v48 = *v37 + 512;
          v53 = *v37 + 8 * (v36 - (v52 << 6));
LABEL_17:
          v77 = v30;
          v44 = v37;
          v79 = v31;
          v78 = v45;
          v73 = v28;
          v80 = (_QWORD *)v61;
          v74 = v29;
          v75 = v57;
          v76 = v27;
          v69 = v58;
          v70 = v26;
          v71 = v66;
          v72 = v25;
          sub_2769DA0(v67, (__int64)&v69, &v73, (__int64)&v77);
          a1[7] = v35;
          a1[6] = v53;
          a1[8] = v48;
          a1[9] = (unsigned __int64)v44;
          return;
        }
        v38 = a1[3];
        v39 = (__int64)(v33 - v38) >> 3;
        if ( v32 > v39 )
        {
          v46 = v27;
          v50 = v29;
          v55 = v28;
          v63 = ((v27 - v25 - 1) << 6) + ((__int64)(v28 - v29) >> 3) + ((v66 - v58) >> 3);
          sub_27697E0(a1, v32 - v39);
          v33 = a1[2];
          v38 = a1[3];
          v27 = v46;
          v29 = v50;
          v28 = v55;
          v32 = v63;
          v39 = (__int64)(v33 - v38) >> 3;
        }
        v40 = (_QWORD *)a1[5];
        v62 = a1[4];
        v41 = v39 - v32;
        if ( v41 < 0 )
        {
          v43 = ~((unsigned __int64)~v41 >> 6);
        }
        else
        {
          if ( v41 <= 63 )
          {
            v42 = v33 - 8 * v32;
LABEL_26:
            v69 = v42;
            v49 = v42;
            v75 = v57;
            v72 = v40;
            v77 = v58;
            v71 = v62;
            v79 = v66;
            v54 = v40;
            v70 = v38;
            v73 = v28;
            v74 = v29;
            v76 = v27;
            v78 = v26;
            v80 = v25;
            sub_2D13060(v67, &v77, &v73, &v69);
            a1[3] = v38;
            a1[2] = v49;
            a1[5] = (unsigned __int64)v54;
            a1[4] = v62;
            return;
          }
          v43 = v41 >> 6;
        }
        v40 += v43;
        v38 = *v40;
        v42 = *v40 + 8 * (v41 - (v43 << 6));
        v62 = *v40 + 512LL;
        goto LABEL_26;
      }
      v24 = v23 >> 6;
    }
    v25 = &v10[v24];
    v26 = *v25;
    v66 = *v25 + 512LL;
    v58 = *v25 + 8 * (v23 - (v24 << 6));
    goto LABEL_10;
  }
  v17 = a2[8];
  v72 = v10;
  v69 = v15;
  v73 = v14;
  v78 = v59;
  v74 = v11;
  v75 = v17;
  v76 = v9;
  v70 = v13;
  v77 = v6;
  v79 = v5;
  v80 = (_QWORD *)v4;
  v71 = v12;
  sub_2769DA0(v67, (__int64)&v69, &v73, (__int64)&v77);
  v18 = v68;
  v19 = v67[1];
  v65 = v67[0];
  v20 = (unsigned __int64 *)(v68 + 8);
  v60 = v67[2];
  v21 = a1[9] + 8;
  if ( v21 > v68 + 8 )
  {
    do
    {
      v22 = *v20++;
      j_j___libc_free_0(v22);
    }
    while ( v21 > (unsigned __int64)v20 );
  }
  a1[7] = v19;
  a1[9] = v18;
  a1[6] = v65;
  a1[8] = v60;
}

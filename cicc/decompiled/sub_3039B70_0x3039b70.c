// Function: sub_3039B70
// Address: 0x3039b70
//
__int64 __fastcall sub_3039B70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r14
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r15
  unsigned int v13; // r13d
  unsigned __int16 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  int v17; // eax
  int v18; // r8d
  int v19; // eax
  __int64 v20; // rsi
  int v21; // edi
  __int128 v22; // rax
  int v23; // r9d
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rbx
  __int128 v27; // rax
  int v28; // r9d
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r15
  __int64 v32; // r14
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdi
  unsigned __int16 *v37; // rdx
  int v38; // r9d
  __int64 v39; // r12
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r15
  __int64 v44; // r14
  __int128 v45; // rax
  int v46; // r9d
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rdx
  int v50; // r9d
  __int64 v51; // rax
  __int64 v52; // rdx
  int v53; // edx
  __int128 v54; // [rsp-30h] [rbp-F0h]
  __int128 v55; // [rsp-20h] [rbp-E0h]
  __int128 v56; // [rsp-20h] [rbp-E0h]
  __int128 v57; // [rsp-20h] [rbp-E0h]
  __int128 v58; // [rsp-20h] [rbp-E0h]
  __int128 v59; // [rsp-20h] [rbp-E0h]
  __int128 v60; // [rsp-10h] [rbp-D0h]
  __int64 v62; // [rsp+10h] [rbp-B0h]
  int v63; // [rsp+10h] [rbp-B0h]
  __int128 v64; // [rsp+10h] [rbp-B0h]
  __int16 v65; // [rsp+22h] [rbp-9Eh]
  __int64 v66; // [rsp+28h] [rbp-98h]
  int v67; // [rsp+28h] [rbp-98h]
  int v68; // [rsp+28h] [rbp-98h]
  __int64 v69; // [rsp+28h] [rbp-98h]
  __int128 v70; // [rsp+30h] [rbp-90h]
  __int64 v71; // [rsp+40h] [rbp-80h] BYREF
  int v72; // [rsp+48h] [rbp-78h]
  __int16 v73; // [rsp+50h] [rbp-70h] BYREF
  __int64 v74; // [rsp+58h] [rbp-68h]
  __int64 v75; // [rsp+60h] [rbp-60h] BYREF
  __int64 v76; // [rsp+68h] [rbp-58h]
  __int64 v77; // [rsp+70h] [rbp-50h]
  __int64 v78; // [rsp+78h] [rbp-48h]
  __int64 v79; // [rsp+80h] [rbp-40h]
  __int64 v80; // [rsp+88h] [rbp-38h]

  v4 = a3;
  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *(_QWORD *)v7;
  v10 = *(_QWORD *)v7;
  v11 = *(_QWORD *)(v7 + 40);
  v71 = v8;
  v12 = *(_QWORD *)(v7 + 8);
  v13 = *(_DWORD *)(v7 + 8);
  v70 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
  if ( v8 )
  {
    v62 = v11;
    v66 = v10;
    sub_B96E90((__int64)&v71, v8, 1);
    v4 = a3;
    v11 = v62;
    v10 = v66;
  }
  v72 = *(_DWORD *)(a2 + 72);
  v14 = (unsigned __int16 *)(*(_QWORD *)(v10 + 48) + 16LL * v13);
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  v73 = v15;
  v74 = v16;
  if ( (_WORD)v15 == 37 )
  {
    v75 = v9;
    v76 = v12;
    v41 = sub_3400BD0(a4, 8, (unsigned int)&v71, 7, 0, 0, 0);
    v43 = v42;
    v44 = v41;
    *(_QWORD *)&v45 = sub_33FB310(a4, v70, *((_QWORD *)&v70 + 1), &v71, 7, 0);
    *((_QWORD *)&v58 + 1) = v43;
    *(_QWORD *)&v58 = v44;
    v47 = sub_3406EB0(a4, 58, (unsigned int)&v71, 7, 0, v46, v45, v58);
    v78 = v48;
    v77 = v47;
    v79 = sub_3400BD0(a4, 8, (unsigned int)&v71, 7, 0, 0, 0);
    v80 = v49;
    *((_QWORD *)&v59 + 1) = 3;
    *(_QWORD *)&v59 = &v75;
    v51 = sub_33FC220(a4, 535, (unsigned int)&v71, 7, 0, v50, v59);
    v39 = sub_33FAFB0(a4, v51, v52, &v71, **(unsigned __int16 **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
  }
  else
  {
    v17 = *(_DWORD *)(v11 + 24);
    if ( v17 == 11 || v17 == 35 )
    {
      v39 = a2;
    }
    else
    {
      if ( (_WORD)v15 )
      {
        v18 = 0;
        LOWORD(v19) = word_4456580[(int)v15 - 1];
      }
      else
      {
        v19 = sub_3009970((__int64)&v73, v8, v15, v11, v4);
        v65 = HIWORD(v19);
        v18 = v53;
      }
      HIWORD(v21) = v65;
      v20 = *(_QWORD *)(a2 + 80);
      LOWORD(v21) = v19;
      v75 = v20;
      if ( v20 )
      {
        v67 = v18;
        sub_B96E90((__int64)&v75, v20, 1);
        v18 = v67;
      }
      v68 = v18;
      LODWORD(v76) = *(_DWORD *)(a2 + 72);
      *(_QWORD *)&v22 = sub_3400D50(a4, 0, &v75, 0);
      *((_QWORD *)&v55 + 1) = v12;
      *(_QWORD *)&v55 = v9;
      v63 = v68;
      v24 = sub_3406EB0(a4, 158, (unsigned int)&v75, v21, v68, v23, v55, v22);
      v69 = v25;
      v26 = v24;
      *(_QWORD *)&v27 = sub_3400D50(a4, 1, &v75, 0);
      *((_QWORD *)&v56 + 1) = v12;
      *(_QWORD *)&v56 = v9;
      v29 = sub_3406EB0(a4, 158, (unsigned int)&v75, v21, v63, v28, v56, v27);
      v31 = v30;
      v32 = v29;
      *(_QWORD *)&v64 = sub_3400D50(a4, 0, &v75, 0);
      *((_QWORD *)&v64 + 1) = v33;
      v34 = sub_33ED040(a4, 17);
      v36 = v35;
      v37 = (unsigned __int16 *)(*(_QWORD *)(v26 + 48) + 16LL * (unsigned int)v69);
      *((_QWORD *)&v60 + 1) = v36;
      *(_QWORD *)&v60 = v34;
      *((_QWORD *)&v57 + 1) = v31;
      *(_QWORD *)&v57 = v32;
      *((_QWORD *)&v54 + 1) = v69;
      *(_QWORD *)&v54 = v26;
      v39 = sub_33FC1D0(a4, 207, (unsigned int)&v75, *v37, *((_QWORD *)v37 + 1), v38, v70, v64, v54, v57, v60);
      if ( v75 )
        sub_B91220((__int64)&v75, v75);
    }
  }
  if ( v71 )
    sub_B91220((__int64)&v71, v71);
  return v39;
}

// Function: sub_3394BA0
// Address: 0x3394ba0
//
__int64 __fastcall sub_3394BA0(__int64 a1, __int64 a2)
{
  __int16 v3; // r13
  __int64 v4; // r15
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // r10d
  __int64 v14; // r9
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  int v20; // eax
  int v21; // edx
  __int64 v22; // r9
  int v23; // r8d
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned __int64 v26; // r13
  __int128 v27; // rax
  __int64 v28; // r12
  int v29; // edx
  _QWORD *v30; // rax
  __int64 v31; // rsi
  __int64 result; // rax
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // r11
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // r9
  unsigned int v39; // r10d
  unsigned int v40; // edx
  int v41; // edx
  __int64 v42; // rax
  __int64 v43; // r15
  __int64 v44; // rsi
  unsigned int v45; // edx
  __int128 v46; // [rsp-30h] [rbp-110h]
  __int128 v47; // [rsp-20h] [rbp-100h]
  int v48; // [rsp+8h] [rbp-D8h]
  __int64 v49; // [rsp+10h] [rbp-D0h]
  int v50; // [rsp+10h] [rbp-D0h]
  unsigned int v51; // [rsp+10h] [rbp-D0h]
  unsigned int v52; // [rsp+10h] [rbp-D0h]
  __int64 v53; // [rsp+18h] [rbp-C8h]
  __int64 *v54; // [rsp+18h] [rbp-C8h]
  int v55; // [rsp+18h] [rbp-C8h]
  unsigned int v56; // [rsp+18h] [rbp-C8h]
  __int64 v57; // [rsp+18h] [rbp-C8h]
  __int64 v58; // [rsp+18h] [rbp-C8h]
  unsigned int v59; // [rsp+24h] [rbp-BCh]
  __int64 v60; // [rsp+28h] [rbp-B8h]
  __int64 v61; // [rsp+28h] [rbp-B8h]
  __int64 v62; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v63; // [rsp+38h] [rbp-A8h]
  int v64; // [rsp+38h] [rbp-A8h]
  unsigned int v65; // [rsp+40h] [rbp-A0h]
  __int64 v66; // [rsp+40h] [rbp-A0h]
  __int64 v67; // [rsp+78h] [rbp-68h] BYREF
  __int64 v68; // [rsp+80h] [rbp-60h] BYREF
  int v69; // [rsp+88h] [rbp-58h]
  __int64 v70; // [rsp+90h] [rbp-50h] BYREF
  int v71; // [rsp+98h] [rbp-48h]
  __int64 v72; // [rsp+A0h] [rbp-40h]

  v3 = *(_WORD *)(a2 + 2);
  v4 = sub_338B750(a1, *(_QWORD *)(a2 - 64));
  v6 = v5;
  v60 = v4;
  v65 = v5;
  v7 = sub_338B750(a1, *(_QWORD *)(a2 - 32));
  v63 = v8;
  v62 = v7;
  v59 = sub_34B9220(v3 & 0x3F);
  v9 = *(_QWORD *)(a1 + 864);
  v10 = *(_QWORD *)(v9 + 16);
  v53 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL);
  v11 = sub_2E79000(*(__int64 **)(v9 + 40));
  v13 = sub_336EEB0(v10, v11, v53, 0);
  v14 = v12;
  v15 = *(_QWORD *)(v4 + 48) + 16LL * (unsigned int)v6;
  if ( *(_WORD *)v15 != (_WORD)v13 || *(_QWORD *)(v15 + 8) != v12 && !*(_WORD *)v15 )
  {
    v33 = *(_DWORD *)(a1 + 848);
    v34 = *(_QWORD *)a1;
    v70 = 0;
    v35 = *(_QWORD *)(a1 + 864);
    v71 = v33;
    if ( v34 )
    {
      if ( &v70 != (__int64 *)(v34 + 48) )
      {
        v36 = *(_QWORD *)(v34 + 48);
        v70 = v36;
        if ( v36 )
        {
          v56 = v13;
          v61 = v14;
          v66 = v35;
          sub_B96E90((__int64)&v70, v36, 1);
          v13 = v56;
          v14 = v61;
          v35 = v66;
        }
      }
    }
    v51 = v13;
    v57 = v14;
    v37 = sub_33FB4C0(v35, v4, v6, &v70, v13, v14);
    v38 = v57;
    v60 = v37;
    v39 = v51;
    v65 = v40;
    if ( v70 )
    {
      sub_B91220((__int64)&v70, v70);
      v39 = v51;
      v38 = v57;
    }
    v41 = *(_DWORD *)(a1 + 848);
    v42 = *(_QWORD *)a1;
    v70 = 0;
    v43 = *(_QWORD *)(a1 + 864);
    v71 = v41;
    if ( v42 )
    {
      if ( &v70 != (__int64 *)(v42 + 48) )
      {
        v44 = *(_QWORD *)(v42 + 48);
        v70 = v44;
        if ( v44 )
        {
          v52 = v39;
          v58 = v38;
          sub_B96E90((__int64)&v70, v44, 1);
          v39 = v52;
          v38 = v58;
        }
      }
    }
    v62 = sub_33FB4C0(v43, v62, v63, &v70, v39, v38);
    v63 = v45 | v63 & 0xFFFFFFFF00000000LL;
    if ( v70 )
      sub_B91220((__int64)&v70, v70);
  }
  v16 = ((*(_BYTE *)(a2 + 1) & 2) != 0) << 14;
  v70 = *(_QWORD *)(a1 + 864);
  v71 = v16;
  v72 = *(_QWORD *)(v70 + 1024);
  *(_QWORD *)(v70 + 1024) = &v70;
  v17 = *(_QWORD *)(a1 + 864);
  v18 = *(_QWORD *)(v17 + 16);
  v54 = *(__int64 **)(a2 + 8);
  v19 = sub_2E79000(*(__int64 **)(v17 + 40));
  v20 = sub_2D5BAE0(v18, v19, v54, 0);
  v68 = 0;
  v22 = *(_QWORD *)(a1 + 864);
  v55 = v20;
  v23 = v21;
  v24 = *(_QWORD *)a1;
  v69 = *(_DWORD *)(a1 + 848);
  if ( v24 )
  {
    if ( &v68 != (__int64 *)(v24 + 48) )
    {
      v25 = *(_QWORD *)(v24 + 48);
      v68 = v25;
      if ( v25 )
      {
        v48 = v21;
        v49 = v22;
        sub_B96E90((__int64)&v68, v25, 1);
        v23 = v48;
        v22 = v49;
      }
    }
  }
  v26 = v63;
  v50 = v23;
  v64 = v22;
  *(_QWORD *)&v27 = sub_33ED040(v22, v59);
  *((_QWORD *)&v47 + 1) = v26;
  *(_QWORD *)&v47 = v62;
  *((_QWORD *)&v46 + 1) = v65 | v6 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v46 = v60;
  v28 = sub_340F900(v64, 208, (unsigned int)&v68, v55, v50, v64, v46, v47, v27);
  LODWORD(v26) = v29;
  v67 = a2;
  v30 = sub_337DC20(a1 + 8, &v67);
  *v30 = v28;
  v31 = v68;
  *((_DWORD *)v30 + 2) = v26;
  if ( v31 )
    sub_B91220((__int64)&v68, v31);
  result = v70;
  *(_QWORD *)(v70 + 1024) = v72;
  return result;
}

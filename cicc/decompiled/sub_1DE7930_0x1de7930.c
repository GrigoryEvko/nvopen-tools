// Function: sub_1DE7930
// Address: 0x1de7930
//
__int64 __fastcall sub_1DE7930(__int64 a1, __int64 a2, __int64 a3, int a4, _QWORD **a5, __int64 a6)
{
  int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // r12d
  __int64 v15; // r15
  __int64 *v16; // rbx
  __int64 v17; // r13
  unsigned int v18; // eax
  __int64 v19; // r13
  __int64 *v20; // r12
  __int64 v21; // rdi
  int v22; // edx
  unsigned int v23; // eax
  __int64 v24; // r8
  unsigned __int64 v25; // rax
  __int64 v26; // rax
  int v27; // eax
  int v28; // r13d
  unsigned __int64 *v29; // rdi
  __int64 v30; // r14
  __int64 v31; // r12
  unsigned __int64 *v32; // rdi
  __int64 v33; // rbx
  __int64 v34; // r13
  unsigned __int64 v35; // r12
  int v37; // ebx
  unsigned int v38; // esi
  __int64 v39; // r15
  __int64 v40; // rax
  int v41; // r8d
  unsigned __int64 *v42; // rdi
  __int64 v43; // rax
  unsigned __int64 *v44; // rdi
  __int64 v45; // r12
  __int64 v46; // rax
  __int64 v47; // rsi
  int v48; // r9d
  __int64 *v49; // rax
  char v50; // al
  unsigned __int64 *v51; // rdi
  __int64 v52; // rax
  unsigned __int64 *v53; // rdi
  __int64 v54; // r13
  __int64 v55; // rax
  unsigned int v56; // [rsp+Ch] [rbp-114h]
  unsigned __int64 v58; // [rsp+18h] [rbp-108h]
  unsigned int v59; // [rsp+30h] [rbp-F0h]
  int v60; // [rsp+34h] [rbp-ECh]
  unsigned __int64 v62; // [rsp+48h] [rbp-D8h]
  __int64 v63; // [rsp+50h] [rbp-D0h]
  __int64 *v64; // [rsp+58h] [rbp-C8h]
  __int64 *v65; // [rsp+58h] [rbp-C8h]
  int v66; // [rsp+6Ch] [rbp-B4h] BYREF
  __int64 v67; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v68; // [rsp+78h] [rbp-A8h] BYREF
  __int64 v69; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v70; // [rsp+88h] [rbp-98h] BYREF
  unsigned __int64 v71; // [rsp+90h] [rbp-90h] BYREF
  unsigned __int64 v72; // [rsp+98h] [rbp-88h] BYREF
  unsigned __int64 v73; // [rsp+A0h] [rbp-80h] BYREF
  unsigned __int64 v74; // [rsp+A8h] [rbp-78h] BYREF
  __int64 v75; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v76; // [rsp+B8h] [rbp-68h] BYREF
  __int64 *v77; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v78; // [rsp+C8h] [rbp-58h]
  _BYTE v79[80]; // [rsp+D0h] [rbp-50h] BYREF

  v77 = (__int64 *)v79;
  v67 = 0;
  v78 = 0x400000000LL;
  v59 = sub_1DE7010(a1, a3, a5, a6, (__int64)&v77);
  v9 = sub_1DF1780(*(_QWORD *)(a1 + 560), a2, a3);
  v10 = sub_20D7490(*(_QWORD *)(a1 + 568), a2);
  v11 = *(_QWORD *)(a1 + 568);
  v68 = v10;
  v69 = sub_20D7490(v11, a3);
  v70 = sub_16AF500(&v68, v9);
  v12 = sub_16AF500(&v68, a4);
  v13 = *(_QWORD *)(a1 + 568);
  v71 = v12;
  v58 = sub_20D7520(v13);
  if ( (_DWORD)v78 )
  {
    v14 = 0;
    v15 = a1;
    v16 = v77;
    v64 = &v77[(unsigned int)v78];
    do
    {
      v17 = *v16;
      v18 = sub_1DF1780(*(_QWORD *)(v15 + 560), a3, *v16);
      if ( v14 < v18 )
        v14 = v18;
      if ( (unsigned __int8)sub_1E5EBB0(*(_QWORD *)(*(_QWORD *)(v15 + 608) + 232LL), v17, a3) )
      {
        v67 = v17;
        v19 = v15;
        goto LABEL_9;
      }
      ++v16;
    }
    while ( v64 != v16 );
    v19 = v15;
LABEL_9:
    v65 = *(__int64 **)(a3 + 72);
    if ( v65 != *(__int64 **)(a3 + 64) )
    {
      v56 = v14;
      v20 = *(__int64 **)(a3 + 64);
      v62 = 0;
      while ( 1 )
      {
        v26 = *v20;
        v75 = v26;
        if ( v26 != a3 && v26 != a2 && (_QWORD **)sub_1DE4FA0(v19 + 888, &v75)[1] != a5 )
        {
          if ( !a6 )
            goto LABEL_13;
          if ( (*(_BYTE *)(a6 + 8) & 1) != 0 )
          {
            v21 = a6 + 16;
            v22 = 15;
            goto LABEL_12;
          }
          v27 = *(_DWORD *)(a6 + 24);
          v21 = *(_QWORD *)(a6 + 16);
          if ( v27 )
          {
            v22 = v27 - 1;
LABEL_12:
            v23 = v22 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
            v24 = *(_QWORD *)(v21 + 8LL * v23);
            if ( v75 != v24 )
            {
              v48 = 1;
              while ( v24 != -8 )
              {
                v23 = v22 & (v48 + v23);
                v24 = *(_QWORD *)(v21 + 8LL * v23);
                if ( v75 == v24 )
                  goto LABEL_13;
                ++v48;
              }
              goto LABEL_16;
            }
LABEL_13:
            v60 = sub_1DF1780(*(_QWORD *)(v19 + 560), v75, a3);
            v76 = sub_20D7490(*(_QWORD *)(v19 + 568), v75);
            v25 = sub_16AF500(&v76, v60);
            if ( v62 >= v25 )
              v25 = v62;
            v62 = v25;
          }
        }
LABEL_16:
        if ( v65 == ++v20 )
        {
          v14 = v56;
          goto LABEL_25;
        }
      }
    }
    v62 = 0;
LABEL_25:
    v72 = v62;
    if ( v67 && sub_1DD6970(a3, v67) )
    {
      v37 = 0;
      v38 = sub_1DF1780(*(_QWORD *)(v19 + 560), a3, v67);
      if ( v38 <= v59 )
        v37 = v59 - v38;
      v39 = sub_16AF500(&v69, v38);
      v63 = sub_16AF500(&v69, v37);
      v40 = sub_16AF5D0(&v69, v72);
      v41 = v38;
      v73 = v40;
      if ( v38 <= v59 >> 1
        || (v49 = sub_1DE4FA0(v19 + 888, &v67),
            v50 = sub_1DE73C0(v19, a3, v67, v49[1], v38, (__int64)a5, a6),
            v41 = v38,
            v50) )
      {
        v42 = &v73;
        if ( v73 <= v72 )
          v42 = &v72;
        v43 = sub_16AF500((__int64 *)v42, v41);
        v44 = &v72;
        v45 = v43;
        if ( v72 > v73 )
          v44 = &v73;
        v46 = sub_16AF500((__int64 *)v44, v59);
        v74 = sub_16AF590((__int64 *)&v71, v46);
        v47 = v39;
        v35 = sub_16AF590((__int64 *)&v74, v45);
      }
      else
      {
        v51 = &v72;
        if ( v72 > v73 )
          v51 = &v73;
        v52 = sub_16AF500((__int64 *)v51, v38);
        v53 = &v72;
        v54 = v52;
        if ( v73 > v72 )
          v53 = &v73;
        v55 = sub_16AF500((__int64 *)v53, v37);
        v74 = sub_16AF590((__int64 *)&v71, v55);
        v47 = v63;
        v35 = sub_16AF590((__int64 *)&v74, v54);
      }
      v75 = sub_16AF590(&v70, v47);
      sub_16AF710(&v66, dword_4FC4CA0, 0x64u);
      v76 = sub_16AF5D0(&v75, v35);
      LOBYTE(v35) = v58 <= sub_16AF550(&v76, v66);
    }
    else
    {
      v28 = v59 - v14;
      if ( v14 > v59 )
        v28 = 0;
      v74 = sub_16AF5D0(&v69, v72);
      v29 = &v72;
      v30 = sub_16AF500(&v69, v28);
      if ( v72 > v74 )
        v29 = &v74;
      v31 = sub_16AF500((__int64 *)v29, v14);
      v32 = &v72;
      v33 = sub_16AF590(&v70, v30);
      if ( v74 > v72 )
        v32 = &v74;
      v34 = sub_16AF500((__int64 *)v32, v28);
      v76 = sub_16AF590((__int64 *)&v71, v31);
      v35 = sub_16AF590(&v76, v34);
      v75 = v33;
      sub_16AF710(&v73, dword_4FC4CA0, 0x64u);
      v76 = sub_16AF5D0(&v75, v35);
      LOBYTE(v35) = v58 <= sub_16AF550(&v76, v73);
    }
  }
  else
  {
    v35 = v71;
    v75 = v70;
    sub_16AF710(&v74, dword_4FC4CA0, 0x64u);
    v76 = sub_16AF5D0(&v75, v35);
    LOBYTE(v35) = v58 <= sub_16AF550(&v76, v74);
  }
  if ( v77 != (__int64 *)v79 )
    _libc_free((unsigned __int64)v77);
  return (unsigned int)v35;
}

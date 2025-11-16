// Function: sub_12DE8F0
// Address: 0x12de8f0
//
__int64 __fastcall sub_12DE8F0(__int64 a1, int a2, __int64 a3)
{
  _DWORD *v6; // rax
  _BYTE *v7; // rsi
  int *v8; // rax
  int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 v12; // rdi
  _BYTE *v13; // rdi
  unsigned int *v14; // rax
  char *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 result; // rax
  __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rsi
  _DWORD *v30; // rax
  int v31; // eax
  unsigned int *v32; // rax
  _DWORD *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rsi
  unsigned int v48; // eax
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // rsi
  char *v60; // [rsp-10h] [rbp-60h]
  _BYTE *v61; // [rsp-10h] [rbp-60h]
  __int64 v62; // [rsp-10h] [rbp-60h]
  __int64 v63; // [rsp-8h] [rbp-58h]
  __int64 v64; // [rsp-8h] [rbp-58h]
  _BYTE *v65; // [rsp-8h] [rbp-58h]
  _BYTE v66[16]; // [rsp+0h] [rbp-50h] BYREF
  __int64 (__fastcall *v67)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-40h]

  v6 = (_DWORD *)sub_1C42D70(4, 4);
  *v6 = a2;
  v7 = v6;
  sub_16D40E0(qword_4FBB410, v6);
  v8 = (int *)sub_16D40F0(qword_4FBB3B0);
  if ( v8 )
    v9 = *v8;
  else
    v9 = qword_4FBB3B0[2];
  if ( v9 )
  {
    v30 = (_DWORD *)sub_16D40F0(qword_4FBB3B0);
    v31 = v30 ? *v30 : LODWORD(qword_4FBB3B0[2]);
    if ( v31 != 1 )
    {
      v13 = qword_4FBB3B0;
      v14 = (unsigned int *)sub_16D40F0(qword_4FBB3B0);
      if ( !v14 )
        goto LABEL_196;
LABEL_35:
      result = *v14;
      if ( !(_DWORD)result )
        goto LABEL_36;
      goto LABEL_197;
    }
  }
  if ( a2 == 3 && !BYTE4(qword_4FBB370[2]) )
  {
    v33 = (_DWORD *)sub_1C42D70(4, 4);
    *v33 = 6;
    sub_16D40E0(qword_4FBB370, v33);
  }
  if ( *(_BYTE *)(a3 + 2000) )
  {
    if ( *(_BYTE *)(a3 + 2600) )
      goto LABEL_7;
  }
  else
  {
    v24 = sub_1CB4E40(1);
    sub_12DE0B0(a1, v24, 0, 0);
    if ( *(_BYTE *)(a3 + 2600) )
      goto LABEL_181;
  }
  v25 = ((__int64 (*)(void))sub_1A223D0)();
  sub_12DE0B0(a1, v25, 0, 0);
LABEL_181:
  if ( !*(_BYTE *)(a3 + 2000) )
  {
    v26 = sub_1CB4E40(1);
    sub_12DE0B0(a1, v26, 0, 1);
    if ( !*(_BYTE *)(a3 + 3488) )
      goto LABEL_8;
    goto LABEL_183;
  }
LABEL_7:
  if ( !*(_BYTE *)(a3 + 3488) )
    goto LABEL_8;
LABEL_183:
  v27 = sub_18E4A00();
  sub_12DE0B0(a1, v27, 0, 0);
  v28 = sub_1C98160(0);
  sub_12DE0B0(a1, v28, 1u, 0);
  if ( *(_BYTE *)(a3 + 3160) && !*(_BYTE *)(a3 + 1080) )
  {
    v29 = sub_17060B0(1, 0);
    sub_12DE0B0(a1, v29, 0, 0);
  }
LABEL_8:
  if ( !*(_BYTE *)(a3 + 600) )
  {
    v34 = sub_12D4560();
    sub_12DE0B0(a1, v34, 0, 0);
  }
  if ( *(_BYTE *)(a3 + 3200) )
  {
    if ( !*(_BYTE *)(a3 + 920) )
    {
      v43 = sub_185D600();
      sub_12DE0B0(a1, v43, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 880) )
    {
      v42 = ((__int64 (*)(void))sub_1857160)();
      sub_12DE0B0(a1, v42, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 1120) )
    {
      v45 = sub_18A3430();
      sub_12DE0B0(a1, v45, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 720) )
    {
      v10 = sub_1842BC0();
      sub_12DE0B0(a1, v10, 0, 0);
    }
  }
  if ( *(_BYTE *)(a3 + 1080) )
  {
    if ( *(_BYTE *)(a3 + 600) )
      goto LABEL_21;
  }
  else
  {
    v35 = sub_17060B0(1, 0);
    sub_12DE0B0(a1, v35, 0, 0);
    if ( *(_BYTE *)(a3 + 600) )
      goto LABEL_21;
  }
  v36 = sub_12D4560();
  sub_12DE0B0(a1, v36, 0, 0);
LABEL_21:
  if ( *(_BYTE *)(a3 + 3200) )
  {
    if ( !*(_BYTE *)(a3 + 2160) )
    {
      v47 = sub_18A3090();
      sub_12DE0B0(a1, v47, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 1960) )
    {
      v11 = sub_184CD60();
      sub_12DE0B0(a1, v11, 0, 0);
    }
  }
  if ( a2 != 1 && !*(_BYTE *)(a3 + 1040) && !*(_BYTE *)(a3 + 1200) )
  {
    v21 = sub_190BB10(1, 0);
    sub_12DE0B0(a1, v21, 0, 0);
    if ( !*(_BYTE *)(a3 + 1160) )
    {
      v59 = sub_1952F90(0xFFFFFFFFLL);
      sub_12DE0B0(a1, v59, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 600) )
    {
      v58 = sub_12D4560();
      sub_12DE0B0(a1, v58, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 1080) )
    {
      v22 = sub_17060B0(1, 0);
      sub_12DE0B0(a1, v22, 0, 0);
    }
  }
  v12 = 0;
  if ( *(_BYTE *)(a3 + 3704) )
  {
    if ( *(_BYTE *)(a3 + 2880) && !*(_BYTE *)(a3 + 1240) )
    {
      v20 = sub_195E880(0);
      sub_12DE0B0(a1, v20, 0, 0);
    }
    v12 = 1;
  }
  v7 = (_BYTE *)sub_1C8A4D0(v12);
  sub_12DE0B0(a1, (__int64)v7, 0, 0);
  if ( a2 != 1 && !*(_BYTE *)(a3 + 1040) )
  {
    v7 = (_BYTE *)sub_1869C50(1, 0, 1);
    sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  v13 = qword_4FBB3B0;
  v14 = (unsigned int *)sub_16D40F0(qword_4FBB3B0);
  if ( v14 )
    goto LABEL_35;
LABEL_196:
  result = LODWORD(qword_4FBB3B0[2]);
  if ( !(_DWORD)result )
    goto LABEL_36;
LABEL_197:
  v13 = qword_4FBB3B0;
  v32 = (unsigned int *)sub_16D40F0(qword_4FBB3B0);
  if ( v32 )
    result = *v32;
  else
    result = LODWORD(qword_4FBB3B0[2]);
  if ( (_DWORD)result != 2 )
    return result;
LABEL_36:
  if ( a2 == 3 && !*(_BYTE *)(a3 + 320) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1833EB0(3);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 2360) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1CC3990();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( *(_BYTE *)(a3 + 3040) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_18EEA90();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 600) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_12D4560();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 2600) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)((__int64 (*)(void))sub_1A223D0)();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    if ( *(_BYTE *)(a3 + 2000) )
      goto LABEL_45;
LABEL_218:
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1CB4E40(1);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    goto LABEL_45;
  }
  if ( !*(_BYTE *)(a3 + 2000) )
    goto LABEL_218;
LABEL_45:
  if ( !*(_BYTE *)(a3 + 440) && !*(_BYTE *)(a3 + 480) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1C4B6F0();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( *(_BYTE *)(a3 + 3160) && !*(_BYTE *)(a3 + 1080) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_17060B0(1, 0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( *(_BYTE *)(a3 + 2720) )
  {
    if ( *(_BYTE *)(a3 + 600) )
      goto LABEL_51;
  }
  else
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1A7A9F0();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    if ( *(_BYTE *)(a3 + 600) )
    {
LABEL_51:
      if ( *(_BYTE *)(a3 + 2200) )
        goto LABEL_52;
LABEL_215:
      v37 = sub_1A02540(v13, v7, v15, v16, v17, v18);
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)v37;
      result = sub_12DE0B0(a1, v37, 0, 0);
      if ( *(_BYTE *)(a3 + 1520) )
        goto LABEL_53;
      goto LABEL_216;
    }
  }
  v13 = (_BYTE *)a1;
  v7 = (_BYTE *)sub_12D4560();
  result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  if ( !*(_BYTE *)(a3 + 2200) )
    goto LABEL_215;
LABEL_52:
  if ( *(_BYTE *)(a3 + 1520) )
    goto LABEL_53;
LABEL_216:
  v13 = (_BYTE *)a1;
  v7 = (_BYTE *)sub_198DF00(0xFFFFFFFFLL);
  result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
LABEL_53:
  if ( !*(_BYTE *)(a3 + 1320) )
  {
    if ( !*(_BYTE *)(a3 + 1480) )
    {
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)sub_1C76260();
      result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 1080) )
    {
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)sub_17060B0(1, 0);
      result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 600) )
    {
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)sub_12D4560();
      result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    }
  }
  if ( *(_BYTE *)(a3 + 2880) && !*(_BYTE *)(a3 + 1240) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_195E880(0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 3488)
    || (v13 = (_BYTE *)a1,
        v7 = (_BYTE *)sub_1C98160(0),
        result = sub_12DE0B0(a1, (__int64)v7, 1u, 0),
        !*(_BYTE *)(a3 + 3160)) )
  {
LABEL_62:
    if ( *(_BYTE *)(a3 + 1360) )
      goto LABEL_63;
    goto LABEL_212;
  }
  if ( !*(_BYTE *)(a3 + 1080) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_17060B0(1, 0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    goto LABEL_62;
  }
  if ( *(_BYTE *)(a3 + 1360) )
    goto LABEL_65;
LABEL_212:
  v13 = (_BYTE *)a1;
  v7 = (_BYTE *)sub_19C1680(0, 1);
  result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
LABEL_63:
  if ( !*(_BYTE *)(a3 + 1080) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_17060B0(1, 0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
LABEL_65:
  if ( !*(_BYTE *)(a3 + 1000) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_19401A0();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 1440) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_196A2B0();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 1400) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1968390();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( a2 != 1 )
  {
    v15 = *(char **)(a3 + 4480);
    result = *(unsigned __int8 *)(a3 + 1560);
    if ( *v15 < 0 )
    {
      if ( (_BYTE)result )
        goto LABEL_74;
      v23 = sub_19B73C0(a2, -1, -1, 0, 0, 0, 0);
    }
    else
    {
      if ( (_BYTE)result )
        goto LABEL_74;
      v23 = sub_19B73C0(a2, -1, -1, -1, -1, -1, -1);
    }
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)v23;
    result = sub_12DE0B0(a1, v23, 0, 0);
LABEL_74:
    if ( *(_BYTE *)(a3 + 3160) && !*(_BYTE *)(a3 + 1080) )
    {
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)sub_17060B0(1, 0);
      result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 2760) )
    {
      result = *(_QWORD *)(a3 + 4480);
      if ( *(char *)result >= 0 && !*(_BYTE *)(a3 + 1560) )
      {
        v13 = (_BYTE *)a1;
        v7 = (_BYTE *)sub_19B73C0(a2, -1, -1, 0, 0, -1, -1);
        result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
        v17 = v62;
        v18 = v63;
      }
    }
    goto LABEL_81;
  }
  if ( *(_BYTE *)(a3 + 3160) && !*(_BYTE *)(a3 + 1080) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_17060B0(1, 0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
LABEL_81:
  if ( !*(_BYTE *)(a3 + 600) )
  {
    v67 = 0;
    v39 = sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)v66);
    sub_12DE0B0(a1, v39, 0, 0);
    result = (__int64)v67;
    v7 = v61;
    v13 = v65;
    if ( v67 )
    {
      v7 = v66;
      v13 = v66;
      result = v67(v66, v66, 3);
    }
  }
  if ( !*(_BYTE *)(a3 + 2600) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)((__int64 (*)(void))sub_1A223D0)();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 2000) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1CB4E40(1);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 1080) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_17060B0(1, 0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 960) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_190BB10(0, 0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 1);
  }
  if ( *(_BYTE *)(a3 + 3080) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1922F90();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( *(_BYTE *)(a3 + 2880) && !*(_BYTE *)(a3 + 1240) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_195E880(0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 2320) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1A13320();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 1400) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1968390();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( *(_BYTE *)(a3 + 3160) && !*(_BYTE *)(a3 + 1080) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_17060B0(1, 0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( *(_BYTE *)(a3 + 3040) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_18EEA90();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 760) )
  {
    v38 = sub_18F5480(v13, v7);
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)v38;
    result = sub_12DE0B0(a1, v38, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 280) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_18DEFF0();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 600) )
  {
    v67 = 0;
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1A62BF0(1, 0, 0, 1, 0, 0, 1, (__int64)v66);
    sub_12DE0B0(a1, (__int64)v7, 0, 0);
    result = (__int64)v67;
    v15 = v60;
    v16 = v64;
    if ( v67 )
    {
      v7 = v66;
      v13 = v66;
      result = v67(v66, v66, 3);
    }
  }
  if ( !*(_BYTE *)(a3 + 520) && !*(_BYTE *)(a3 + 560) )
  {
    v41 = sub_1AAC510(v13, v7, v15);
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)v41;
    result = sub_12DE0B0(a1, v41, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 2600) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)((__int64 (*)(void))sub_1A223D0)();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 2000) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1CB4E40(1);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 2680) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1C8E680(*(unsigned __int8 *)(a3 + 3120));
    result = sub_12DE0B0(a1, (__int64)v7, 1u, 0);
  }
  if ( *(_BYTE *)(a3 + 3120) && !*(_BYTE *)(a3 + 2600) )
  {
    v40 = sub_1A223D0(v13, v7, v15);
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)v40;
    result = sub_12DE0B0(a1, v40, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 1080) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_17060B0(1, 0);
    result = sub_12DE0B0(a1, (__int64)v7, 0, 1);
  }
  if ( !*(_BYTE *)(a3 + 2560) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1CC71E0();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( *(_BYTE *)(a3 + 3488) )
  {
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)sub_1C98270(1, *(unsigned __int8 *)(a3 + 2920));
    result = sub_12DE0B0(a1, (__int64)v7, 1u, 0);
    if ( *(_BYTE *)(a3 + 3160) && !*(_BYTE *)(a3 + 1080) )
    {
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)sub_17060B0(1, 0);
      result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    }
    if ( *(_BYTE *)(a3 + 2840) && !*(_BYTE *)(a3 + 1840) )
    {
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)sub_1C6FCA0();
      result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    }
  }
  if ( *(_BYTE *)(a3 + 3200) )
  {
    if ( !*(_BYTE *)(a3 + 2640) )
    {
      v44 = sub_18B1DE0(v13, v7, v15);
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)v44;
      result = sub_12DE0B0(a1, v44, 0, 0);
    }
    if ( a2 == 3 )
    {
      if ( !*(_BYTE *)(a3 + 880) )
      {
        v57 = sub_1857160(v13, v7, v15, v16, v17, v18);
        v13 = (_BYTE *)a1;
        v7 = (_BYTE *)v57;
        result = sub_12DE0B0(a1, v57, 0, 0);
      }
      if ( *(_BYTE *)(a3 + 680) )
        goto LABEL_133;
    }
    else
    {
      if ( a2 == 1 )
        goto LABEL_138;
      if ( *(_BYTE *)(a3 + 680) )
        goto LABEL_135;
    }
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)((__int64 (*)(void))sub_1841180)();
    result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
  }
  if ( a2 == 3 )
  {
LABEL_133:
    if ( !*(_BYTE *)(a3 + 360) )
    {
      v51 = sub_1C46000(v13, v7, v15);
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)v51;
      result = sub_12DE0B0(a1, v51, 0, 0);
    }
    goto LABEL_135;
  }
  if ( a2 != 1 )
  {
LABEL_135:
    if ( *(_BYTE *)(a3 + 3200) && !*(_BYTE *)(a3 + 680) )
    {
      v55 = sub_1841180(v13, v7, v15, v16, v17, v18);
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)v55;
      result = sub_12DE0B0(a1, v55, 0, 0);
    }
  }
LABEL_138:
  if ( !*(_BYTE *)(a3 + 2240) && !*(_BYTE *)(a3 + 2280) )
  {
    v52 = sub_1CBC480(v13, v7, v15);
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)v52;
    result = sub_12DE0B0(a1, v52, 0, 0);
  }
  if ( !*(_BYTE *)(a3 + 2080) && !*(_BYTE *)(a3 + 2120) )
  {
    v50 = sub_1CB73C0(v13, v7, v15);
    v13 = (_BYTE *)a1;
    v7 = (_BYTE *)v50;
    result = sub_12DE0B0(a1, v50, 0, 0);
  }
  if ( *(_BYTE *)(a3 + 3328) )
  {
    if ( !*(_BYTE *)(a3 + 1640) )
    {
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)sub_1C7F370(1, v7, v15);
      result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 2400) )
    {
      v48 = *(_BYTE *)(a3 + 1920) ^ 1;
      BYTE1(v48) = *(_BYTE *)(a3 + 1880) ^ 1;
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)sub_1CC5E00(v48);
      result = sub_12DE0B0(a1, (__int64)v7, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 2440) )
    {
      v49 = sub_1CC60B0(v13, v7, v15);
      v13 = (_BYTE *)a1;
      v7 = (_BYTE *)v49;
      result = sub_12DE0B0(a1, v49, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 2080) && !*(_BYTE *)(a3 + 2120) )
    {
      v56 = sub_1CB73C0(v13, v7, v15);
      result = sub_12DE0B0(a1, v56, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 1080) )
    {
      v54 = sub_17060B0(1, 0);
      result = sub_12DE0B0(a1, v54, 0, 0);
    }
    if ( !*(_BYTE *)(a3 + 1280) )
    {
      v53 = sub_1B7FDF0(3);
      result = sub_12DE0B0(a1, v53, 0, 0);
    }
  }
  if ( *(_BYTE *)(a3 + 3160) )
  {
    if ( !*(_BYTE *)(a3 + 1080) )
    {
      v46 = sub_17060B0(1, 0);
      return sub_12DE0B0(a1, v46, 0, 0);
    }
  }
  return result;
}

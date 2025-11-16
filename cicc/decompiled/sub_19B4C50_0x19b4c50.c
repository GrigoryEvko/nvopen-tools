// Function: sub_19B4C50
// Address: 0x19b4c50
//
__int64 __fastcall sub_19B4C50(
        _QWORD *a1,
        __int64 a2,
        int a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  __int64 *v14; // rdx
  __int64 result; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  _QWORD *v18; // rax
  _DWORD *v19; // rdi
  unsigned __int64 v20; // rsi
  _QWORD *v21; // rax
  _DWORD *v22; // rdi
  __int64 v23; // rdi
  int v24; // eax
  unsigned __int64 v25; // rdx
  _QWORD *v26; // rax
  _DWORD *v27; // rsi
  __int64 v28; // rax
  _DWORD *v29; // r8
  _DWORD *v30; // rdi
  __int64 v31; // rax
  _DWORD *v32; // r8
  _DWORD *v33; // rdi
  __int64 v34; // rax
  int v35; // ecx
  __int64 v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // rsi
  __int64 v39; // rdi
  bool v40; // dl
  __int64 v41; // rax
  _DWORD *v42; // rdx
  unsigned int v43; // esi
  unsigned int v44; // eax
  char v45; // di
  __int64 v46; // rdx
  __int64 v47; // r14
  __int64 v48; // r13
  __int64 v49; // r12
  __int64 i; // rbx
  __int64 v51; // rax
  unsigned int v52; // [rsp+14h] [rbp-22Ch]
  int v53; // [rsp+18h] [rbp-228h]
  int v54; // [rsp+18h] [rbp-228h]
  char v55; // [rsp+23h] [rbp-21Dh]
  int v56; // [rsp+24h] [rbp-21Ch]
  int v57; // [rsp+48h] [rbp-1F8h]
  unsigned int v58; // [rsp+50h] [rbp-1F0h]
  __int64 v59; // [rsp+58h] [rbp-1E8h]
  int v60; // [rsp+58h] [rbp-1E8h]
  __int64 v61; // [rsp+60h] [rbp-1E0h]
  char v62; // [rsp+60h] [rbp-1E0h]
  __int64 v63; // [rsp+68h] [rbp-1D8h]
  int v64; // [rsp+68h] [rbp-1D8h]
  __int64 v65; // [rsp+70h] [rbp-1D0h]
  __int64 v66; // [rsp+70h] [rbp-1D0h]
  unsigned int v68; // [rsp+78h] [rbp-1C8h]
  unsigned int v69; // [rsp+78h] [rbp-1C8h]
  char v70; // [rsp+8Bh] [rbp-1B5h] BYREF
  _BYTE v71[2]; // [rsp+8Ch] [rbp-1B4h] BYREF
  _BYTE v72[2]; // [rsp+8Eh] [rbp-1B2h] BYREF
  int v73; // [rsp+90h] [rbp-1B0h] BYREF
  unsigned int v74; // [rsp+94h] [rbp-1ACh] BYREF
  unsigned int v75; // [rsp+98h] [rbp-1A8h] BYREF
  char v76; // [rsp+9Ch] [rbp-1A4h]
  _DWORD v77[5]; // [rsp+A0h] [rbp-1A0h] BYREF
  unsigned int v78; // [rsp+B4h] [rbp-18Ch]
  unsigned int v79; // [rsp+C8h] [rbp-178h]
  char v80; // [rsp+CDh] [rbp-173h]
  bool v81; // [rsp+CEh] [rbp-172h]
  char v82; // [rsp+D0h] [rbp-170h]
  unsigned __int8 v83; // [rsp+D3h] [rbp-16Dh]
  char v84; // [rsp+D4h] [rbp-16Ch]
  unsigned int v85; // [rsp+D8h] [rbp-168h]
  __int64 v86; // [rsp+E0h] [rbp-160h] BYREF
  _BYTE *v87; // [rsp+E8h] [rbp-158h]
  _BYTE *v88; // [rsp+F0h] [rbp-150h]
  __int64 v89; // [rsp+F8h] [rbp-148h]
  int v90; // [rsp+100h] [rbp-140h]
  _BYTE v91[312]; // [rsp+108h] [rbp-138h] BYREF

  if ( !(unsigned __int8)sub_13FCBF0((__int64)a1) )
    return 0;
  v14 = (__int64 *)a1[1];
  if ( a1[2] - (_QWORD)v14 != 8 )
    return 0;
  v65 = *v14;
  if ( !(unsigned __int8)sub_13FCBF0(*v14) )
    return 0;
  v63 = sub_13FCB50((__int64)a1);
  v61 = sub_13F9E70((__int64)a1);
  v59 = sub_13FCB50(v65);
  v16 = sub_13F9E70(v65);
  if ( v63 != v61 || v59 != v16 )
    return 0;
  BYTE1(v74) = 0;
  BYTE1(v73) = 0;
  v72[1] = 0;
  v71[1] = 0;
  BYTE4(v86) = 0;
  v76 = 0;
  sub_19B6690(
    (unsigned int)v77,
    (_DWORD)a1,
    a4,
    a5,
    a9,
    (unsigned int)&v75,
    (__int64)&v86,
    (__int64)v71,
    (__int64)v72,
    (__int64)&v73,
    (__int64)&v74);
  v17 = sub_16D5D50();
  v18 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v19 = dword_4FA0208;
    do
    {
      if ( v17 > v18[4] )
      {
        v18 = (_QWORD *)v18[3];
      }
      else
      {
        v19 = v18;
        v18 = (_QWORD *)v18[2];
      }
    }
    while ( v18 );
    if ( v19 != dword_4FA0208 && v17 >= *((_QWORD *)v19 + 4) )
    {
      v28 = *((_QWORD *)v19 + 7);
      v29 = v19 + 12;
      if ( v28 )
      {
        v30 = v19 + 12;
        do
        {
          if ( *(_DWORD *)(v28 + 32) < dword_4FB2188 )
          {
            v28 = *(_QWORD *)(v28 + 24);
          }
          else
          {
            v30 = (_DWORD *)v28;
            v28 = *(_QWORD *)(v28 + 16);
          }
        }
        while ( v28 );
        if ( v29 != v30 && dword_4FB2188 >= v30[8] && (int)v30[9] > 0 )
          v84 = byte_4FB2220;
      }
    }
  }
  v20 = sub_16D5D50();
  v21 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v22 = dword_4FA0208;
    do
    {
      if ( v20 > v21[4] )
      {
        v21 = (_QWORD *)v21[3];
      }
      else
      {
        v22 = v21;
        v21 = (_QWORD *)v21[2];
      }
    }
    while ( v21 );
    if ( v22 != dword_4FA0208 && v20 >= *((_QWORD *)v22 + 4) )
    {
      v31 = *((_QWORD *)v22 + 7);
      v32 = v22 + 12;
      if ( v31 )
      {
        v33 = v22 + 12;
        do
        {
          if ( *(_DWORD *)(v31 + 32) < dword_4FB1FC8 )
          {
            v31 = *(_QWORD *)(v31 + 24);
          }
          else
          {
            v33 = (_DWORD *)v31;
            v31 = *(_QWORD *)(v31 + 16);
          }
        }
        while ( v31 );
        if ( v32 != v33 && dword_4FB1FC8 >= v33[8] && (int)v33[9] > 0 )
          v85 = dword_4FB2060;
      }
    }
  }
  if ( !v84 )
    return 0;
  if ( !v85 )
    return 0;
  v23 = sub_13FD000((__int64)a1);
  if ( v23 )
  {
    if ( sub_1AFD990(v23, "llvm.loop.unroll_and_jam.disable", 32) )
      return 0;
  }
  if ( (unsigned __int8)sub_19B4AE0((__int64)a1, "llvm.loop.unroll.", 0x11u)
    && !(unsigned __int8)sub_19B4AE0((__int64)a1, "llvm.loop.unroll_and_jam.", 0x19u) )
  {
    return 0;
  }
  v55 = sub_1B05600(a1, a4, a2, a7);
  if ( !v55 )
    return 0;
  v87 = v91;
  v88 = v91;
  v86 = 0;
  v89 = 32;
  v90 = 0;
  sub_14D04F0((__int64)a1, a6, (__int64)&v86);
  v56 = sub_19B7070(v65, (unsigned int)&v73, (unsigned int)&v70, (unsigned int)v71, a5, (unsigned int)&v86, v79);
  v24 = sub_19B7070((_DWORD)a1, (unsigned int)&v73, (unsigned int)&v70, (unsigned int)v71, a5, (unsigned int)&v86, v79);
  if ( v70 )
    goto LABEL_80;
  v53 = v24;
  v57 = v73;
  if ( v73 || v71[0] )
    goto LABEL_80;
  v58 = sub_1474190(a4, (__int64)a1, v63);
  v52 = sub_147DD60(a4, (__int64)a1, v63);
  v60 = sub_1474190(a4, v65, v59);
  v74 = v58;
  v75 = v52;
  v25 = sub_16D5D50();
  v26 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_60;
  v27 = dword_4FA0208;
  do
  {
    if ( v25 > v26[4] )
    {
      v26 = (_QWORD *)v26[3];
    }
    else
    {
      v27 = v26;
      v26 = (_QWORD *)v26[2];
    }
  }
  while ( v26 );
  if ( v27 == dword_4FA0208 )
    goto LABEL_60;
  if ( v25 < *((_QWORD *)v27 + 4) )
    goto LABEL_60;
  v41 = *((_QWORD *)v27 + 7);
  if ( !v41 )
    goto LABEL_60;
  v42 = v27 + 12;
  do
  {
    if ( *(_DWORD *)(v41 + 32) < dword_4FB20A8 )
    {
      v41 = *(_QWORD *)(v41 + 24);
    }
    else
    {
      v42 = (_DWORD *)v41;
      v41 = *(_QWORD *)(v41 + 16);
    }
  }
  while ( v41 );
  if ( v27 + 12 == v42 || dword_4FB20A8 < v42[8] )
  {
LABEL_60:
    v64 = 0;
  }
  else
  {
    v64 = v42[9];
    if ( v64 > 0 )
    {
      v43 = dword_4FB2140;
      v82 = 1;
      v78 = dword_4FB2140;
      if ( v81
        && v79 + (unsigned int)dword_4FB2140 * (unsigned __int64)(v53 - v79) < v77[0]
        && v79 + (unsigned int)dword_4FB2140 * (unsigned __int64)(v56 - v79) < v85 )
      {
        goto LABEL_97;
      }
    }
  }
  v34 = sub_13FD000((__int64)a1);
  v35 = v53;
  if ( v34 && (v36 = sub_1AFD990(v34, "llvm.loop.unroll_and_jam.count", 30), v35 = v53, v36) )
  {
    v37 = *(_QWORD *)(*(_QWORD *)(v36 + 8 * (1LL - *(unsigned int *)(v36 + 8))) + 136LL);
    v38 = *(_QWORD **)(v37 + 24);
    if ( *(_DWORD *)(v37 + 32) > 0x40u )
      v38 = (_QWORD *)*v38;
    v54 = (int)v38;
    if ( (_DWORD)v38 )
    {
      v78 = (unsigned int)v38;
      v80 = 1;
      v82 = 1;
      if ( (v81 || !(v75 % (unsigned int)v38))
        && v79 + (unsigned int)v38 * (unsigned __int64)(v35 - v79) < v77[0]
        && (v56 - v79) * (unsigned __int64)(unsigned int)v38 + v79 < v85 )
      {
        goto LABEL_122;
      }
    }
  }
  else
  {
    v54 = 0;
  }
  v72[0] = 0;
  if ( (unsigned __int8)sub_19BB5C0(
                          (_DWORD)a1,
                          a5,
                          a2,
                          a3,
                          a4,
                          (unsigned int)&v86,
                          a8,
                          (__int64)&v74,
                          0,
                          (__int64)&v75,
                          v35,
                          (__int64)v77,
                          (__int64)v72) )
    goto LABEL_79;
  v62 = v72[0];
  if ( v72[0] )
    goto LABEL_79;
  v39 = sub_13FD000((__int64)a1);
  if ( v39 )
    v39 = sub_1AFD990(v39, "llvm.loop.unroll_and_jam.enable", 31);
  v40 = v39 != 0 || v64 > 0 || v54 != 0;
  if ( v40 )
  {
    if ( v74 )
      v85 = dword_4FB1F80;
    if ( !v81 )
    {
      if ( v85 <= v79 + (v56 - v79) * (unsigned __int64)v78 )
      {
LABEL_79:
        v78 = 0;
        goto LABEL_80;
      }
      goto LABEL_122;
    }
    v43 = v78;
    v40 = v81;
    v44 = v78;
    if ( !v78 )
    {
LABEL_97:
      v62 = v55;
      goto LABEL_98;
    }
LABEL_107:
    v45 = 0;
    while ( v79 + (v56 - v79) * (unsigned __int64)v44 >= v85 )
    {
      if ( !--v44 )
      {
        v78 = 0;
        goto LABEL_112;
      }
      v45 = v55;
    }
    if ( v45 )
      v78 = v44;
LABEL_112:
    if ( !v40 )
      goto LABEL_113;
LABEL_122:
    v43 = v78;
    goto LABEL_97;
  }
  if ( v81 )
  {
    if ( v60 && (unsigned int)(v56 * v60) < v77[0] )
      goto LABEL_79;
    v44 = v78;
    if ( v78 )
      goto LABEL_107;
  }
  else if ( v79 + v78 * (unsigned __int64)(v56 - v79) >= v85 || v60 && v77[0] > (unsigned int)(v60 * v56) )
  {
    goto LABEL_79;
  }
LABEL_113:
  v46 = *(_QWORD *)(v65 + 32);
  if ( *(_QWORD *)(v65 + 40) - v46 != 8 )
    goto LABEL_79;
  v66 = a6;
  v47 = a2;
  v48 = *(_QWORD *)v46 + 40LL;
  v49 = a4;
  for ( i = *(_QWORD *)(*(_QWORD *)v46 + 48LL); v48 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 8) == 54 )
    {
      v51 = sub_1472610(v49, *(_QWORD *)(i - 48), a1);
      v57 -= !sub_146CEE0(v49, v51, (__int64)a1) - 1;
    }
  }
  a2 = v47;
  a4 = v49;
  a6 = v66;
  if ( !v57 )
    goto LABEL_79;
  v43 = v78;
LABEL_98:
  if ( v43 > 1 )
  {
    if ( v58 && v43 > v58 )
    {
      v78 = v58;
      v43 = v58;
    }
    result = sub_1B07290((_DWORD)a1, v43, v58, v52, v83, a3, a4, a2, a6, a8);
    if ( (_DWORD)result != 2 && v62 )
    {
      v69 = result;
      sub_13FD1C0((__int64)a1);
      result = v69;
    }
    goto LABEL_81;
  }
LABEL_80:
  result = 0;
LABEL_81:
  if ( v88 != v87 )
  {
    v68 = result;
    _libc_free((unsigned __int64)v88);
    return v68;
  }
  return result;
}

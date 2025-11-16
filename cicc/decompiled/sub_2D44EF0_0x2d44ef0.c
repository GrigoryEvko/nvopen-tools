// Function: sub_2D44EF0
// Address: 0x2d44ef0
//
__int64 __fastcall sub_2D44EF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6, unsigned int a7)
{
  __int64 v9; // rbx
  _QWORD **v10; // rax
  __int64 v11; // r15
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // eax
  unsigned __int8 v16; // al
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // eax
  unsigned __int64 v20; // rdx
  char v21; // al
  __int64 v22; // r13
  __int64 v23; // r15
  __int64 v24; // r13
  __int64 v25; // r14
  bool v26; // zf
  __int64 v27; // r15
  __int64 v28; // r13
  __int64 v29; // r14
  __int64 v30; // r13
  __int64 v31; // r15
  __int64 v32; // rdi
  __int64 v33; // r14
  __int64 v34; // r13
  __int64 v36; // r13
  __int64 v37; // r15
  __int64 v38; // r14
  __int64 v39; // r15
  __int64 v40; // r14
  __int64 v41; // rdx
  unsigned int v42; // esi
  __int64 v43; // r14
  __int64 v44; // rbx
  __int64 v45; // r14
  __int64 v46; // rdx
  unsigned int v47; // esi
  __int64 v48; // r13
  __int64 v49; // r14
  __int64 v50; // rdx
  unsigned int v51; // esi
  __int64 v52; // r15
  __int64 v53; // r13
  __int64 v54; // rdx
  unsigned int v55; // esi
  __int64 v56; // r14
  __int64 v57; // r13
  __int64 v58; // rdx
  unsigned int v59; // esi
  __int64 v60; // r15
  __int64 v61; // r14
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // r15
  __int64 v65; // r13
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 v68; // rdx
  int v69; // r13d
  __int64 v70; // r13
  __int64 v71; // rbx
  __int64 v72; // rdx
  unsigned int v73; // esi
  __int64 v74; // [rsp+0h] [rbp-110h]
  _QWORD **v75; // [rsp+10h] [rbp-100h]
  __int64 v77; // [rsp+20h] [rbp-F0h]
  unsigned int v80; // [rsp+30h] [rbp-E0h]
  __int64 v81; // [rsp+48h] [rbp-C8h]
  _QWORD v82[4]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v83; // [rsp+70h] [rbp-A0h]
  _QWORD v84[4]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v85; // [rsp+A0h] [rbp-70h]
  const char *v86; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v87; // [rsp+B8h] [rbp-58h]
  __int16 v88; // [rsp+D0h] [rbp-40h]

  v9 = a2;
  *(_OWORD *)a1 = 0;
  *(_OWORD *)(a1 + 16) = 0;
  *(_OWORD *)(a1 + 32) = 0;
  *(_OWORD *)(a1 + 48) = 0;
  v10 = (_QWORD **)sub_B43CA0(a3);
  v11 = (__int64)(v10 + 39);
  v12 = *v10;
  v75 = v10;
  v13 = sub_9208B0((__int64)(v10 + 39), a4);
  v87 = v14;
  v86 = (const char *)((unsigned __int64)(v13 + 7) >> 3);
  v15 = sub_CA1930(&v86);
  *(_QWORD *)(a1 + 16) = a4;
  v80 = v15;
  v16 = *(_BYTE *)(a4 + 8);
  *(_QWORD *)(a1 + 8) = a4;
  if ( v16 <= 3u || v16 == 5 || (v16 & 0xFD) == 4 || (unsigned __int8)(v16 - 17) <= 1u )
  {
    v17 = sub_BCAE30(a4);
    v87 = v18;
    v86 = (const char *)v17;
    v19 = sub_CA1930(&v86);
    *(_QWORD *)(a1 + 16) = sub_BCD140(v12, v19);
    if ( v80 < a7 )
      goto LABEL_9;
LABEL_3:
    *(_QWORD *)a1 = a4;
LABEL_4:
    *(_QWORD *)(a1 + 24) = a5;
    *(_BYTE *)(a1 + 32) = a6;
    *(_QWORD *)(a1 + 40) = sub_AD64C0(a4, 0, 0);
    *(_QWORD *)(a1 + 48) = sub_AD64C0(a4, -1, 1u);
    return a1;
  }
  if ( v80 >= a7 )
    goto LABEL_3;
LABEL_9:
  v77 = sub_BCD140(v12, 8 * a7);
  *(_QWORD *)a1 = v77;
  if ( a4 == v77 )
    goto LABEL_4;
  _BitScanReverse64(&v20, a7);
  v21 = 63 - (v20 ^ 0x3F);
  LODWORD(v20) = *(_DWORD *)(*(_QWORD *)(a5 + 8) + 8LL);
  v74 = *(_QWORD *)(a5 + 8);
  *(_BYTE *)(a1 + 32) = v21;
  v22 = sub_AE4540(v11, (__int64)v12, (unsigned int)v20 >> 8);
  if ( a7 <= (unsigned __int64)(1LL << a6) )
  {
    *(_QWORD *)(a1 + 24) = a5;
    v25 = sub_AD6530(v22, (__int64)v12);
  }
  else
  {
    v86 = "AlignedAddr";
    v88 = 259;
    BYTE4(v81) = 0;
    v84[0] = a5;
    v82[0] = v74;
    v84[1] = sub_ACD640(v22, ~(unsigned __int64)(a7 - 1), 0);
    v82[1] = v22;
    *(_QWORD *)(a1 + 24) = sub_B33D10(a2, 0x12Bu, (__int64)v82, 2, (int)v84, 2, v81, (__int64)&v86);
    v85 = 257;
    if ( v22 == *(_QWORD *)(a5 + 8) )
    {
      v23 = a5;
    }
    else
    {
      v23 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a2 + 80) + 120LL))(
              *(_QWORD *)(a2 + 80),
              47,
              a5,
              v22);
      if ( !v23 )
      {
        v88 = 257;
        v23 = sub_B51D30(47, a5, v22, (__int64)&v86, 0, 0);
        if ( (unsigned __int8)sub_920620(v23) )
        {
          v68 = *(_QWORD *)(a2 + 96);
          v69 = *(_DWORD *)(a2 + 104);
          if ( v68 )
            sub_B99FD0(v23, 3u, v68);
          sub_B45150(v23, v69);
        }
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
          *(_QWORD *)(a2 + 88),
          v23,
          v84,
          *(_QWORD *)(a2 + 56),
          *(_QWORD *)(a2 + 64));
        if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
        {
          v70 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
          v71 = *(_QWORD *)a2;
          do
          {
            v72 = *(_QWORD *)(v71 + 8);
            v73 = *(_DWORD *)v71;
            v71 += 16;
            sub_B99FD0(v23, v73, v72);
          }
          while ( v70 != v71 );
          v9 = a2;
        }
      }
    }
    v84[0] = "PtrLSB";
    v85 = 259;
    v24 = sub_AD64C0(*(_QWORD *)(v23 + 8), a7 - 1, 0);
    v25 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v9 + 80) + 16LL))(
            *(_QWORD *)(v9 + 80),
            28,
            v23,
            v24);
    if ( !v25 )
    {
      v88 = 257;
      v25 = sub_B504D0(28, v23, v24, (__int64)&v86, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
        *(_QWORD *)(v9 + 88),
        v25,
        v84,
        *(_QWORD *)(v9 + 56),
        *(_QWORD *)(v9 + 64));
      v64 = *(_QWORD *)v9;
      v65 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
      if ( *(_QWORD *)v9 != v65 )
      {
        do
        {
          v66 = *(_QWORD *)(v64 + 8);
          v67 = *(_DWORD *)v64;
          v64 += 16;
          sub_B99FD0(v25, v67, v66);
        }
        while ( v65 != v64 );
      }
    }
  }
  v26 = *((_BYTE *)v75 + 312) == 0;
  v85 = 257;
  if ( v26 )
  {
    v27 = sub_AD64C0(*(_QWORD *)(v25 + 8), 3, 0);
    v28 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v9 + 80) + 32LL))(
            *(_QWORD *)(v9 + 80),
            25,
            v25,
            v27,
            0,
            0);
    if ( !v28 )
    {
      v88 = 257;
      v28 = sub_B504D0(25, v25, v27, (__int64)&v86, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
        *(_QWORD *)(v9 + 88),
        v28,
        v84,
        *(_QWORD *)(v9 + 56),
        *(_QWORD *)(v9 + 64));
      v60 = *(_QWORD *)v9;
      v61 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
      if ( *(_QWORD *)v9 != v61 )
      {
        do
        {
          v62 = *(_QWORD *)(v60 + 8);
          v63 = *(_DWORD *)v60;
          v60 += 16;
          sub_B99FD0(v28, v63, v62);
        }
        while ( v61 != v60 );
      }
    }
  }
  else
  {
    v83 = 257;
    v36 = sub_AD64C0(*(_QWORD *)(v25 + 8), a7 - v80, 0);
    v37 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v9 + 80) + 16LL))(
            *(_QWORD *)(v9 + 80),
            30,
            v25,
            v36);
    if ( !v37 )
    {
      v88 = 257;
      v37 = sub_B504D0(30, v25, v36, (__int64)&v86, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
        *(_QWORD *)(v9 + 88),
        v37,
        v82,
        *(_QWORD *)(v9 + 56),
        *(_QWORD *)(v9 + 64));
      v56 = *(_QWORD *)v9;
      v57 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
      if ( *(_QWORD *)v9 != v57 )
      {
        do
        {
          v58 = *(_QWORD *)(v56 + 8);
          v59 = *(_DWORD *)v56;
          v56 += 16;
          sub_B99FD0(v37, v59, v58);
        }
        while ( v57 != v56 );
      }
    }
    v38 = sub_AD64C0(*(_QWORD *)(v37 + 8), 3, 0);
    v28 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v9 + 80) + 32LL))(
            *(_QWORD *)(v9 + 80),
            25,
            v37,
            v38,
            0,
            0);
    if ( !v28 )
    {
      v88 = 257;
      v28 = sub_B504D0(25, v37, v38, (__int64)&v86, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
        *(_QWORD *)(v9 + 88),
        v28,
        v84,
        *(_QWORD *)(v9 + 56),
        *(_QWORD *)(v9 + 64));
      v39 = *(_QWORD *)v9;
      v40 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
      if ( *(_QWORD *)v9 != v40 )
      {
        do
        {
          v41 = *(_QWORD *)(v39 + 8);
          v42 = *(_DWORD *)v39;
          v39 += 16;
          sub_B99FD0(v28, v42, v41);
        }
        while ( v40 != v39 );
      }
    }
  }
  *(_QWORD *)(a1 + 40) = v28;
  v84[0] = "ShiftAmt";
  v85 = 259;
  if ( v77 == *(_QWORD *)(v28 + 8) )
  {
    v29 = v28;
  }
  else
  {
    v29 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v9 + 80) + 120LL))(
            *(_QWORD *)(v9 + 80),
            38,
            v28);
    if ( !v29 )
    {
      v88 = 257;
      v29 = sub_B51D30(38, v28, v77, (__int64)&v86, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
        *(_QWORD *)(v9 + 88),
        v29,
        v84,
        *(_QWORD *)(v9 + 56),
        *(_QWORD *)(v9 + 64));
      v52 = *(_QWORD *)v9;
      v53 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
      if ( *(_QWORD *)v9 != v53 )
      {
        do
        {
          v54 = *(_QWORD *)(v52 + 8);
          v55 = *(_DWORD *)v52;
          v52 += 16;
          sub_B99FD0(v29, v55, v54);
        }
        while ( v53 != v52 );
      }
    }
  }
  *(_QWORD *)(a1 + 40) = v29;
  v84[0] = "Mask";
  v85 = 259;
  v30 = sub_AD64C0(v77, (1 << (8 * v80)) - 1, 0);
  v31 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v9 + 80) + 32LL))(
          *(_QWORD *)(v9 + 80),
          25,
          v30,
          v29,
          0,
          0);
  if ( !v31 )
  {
    v88 = 257;
    v31 = sub_B504D0(25, v30, v29, (__int64)&v86, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
      *(_QWORD *)(v9 + 88),
      v31,
      v84,
      *(_QWORD *)(v9 + 56),
      *(_QWORD *)(v9 + 64));
    v48 = *(_QWORD *)v9;
    v49 = *(_QWORD *)v9 + 16LL * *(unsigned int *)(v9 + 8);
    if ( *(_QWORD *)v9 != v49 )
    {
      do
      {
        v50 = *(_QWORD *)(v48 + 8);
        v51 = *(_DWORD *)v48;
        v48 += 16;
        sub_B99FD0(v31, v51, v50);
      }
      while ( v49 != v48 );
    }
  }
  v84[0] = "Inv_Mask";
  v85 = 259;
  v32 = *(_QWORD *)(v31 + 8);
  *(_QWORD *)(a1 + 48) = v31;
  v33 = sub_AD62B0(v32);
  v34 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v9 + 80) + 16LL))(
          *(_QWORD *)(v9 + 80),
          30,
          v31,
          v33);
  if ( !v34 )
  {
    v88 = 257;
    v34 = sub_B504D0(30, v31, v33, (__int64)&v86, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
      *(_QWORD *)(v9 + 88),
      v34,
      v84,
      *(_QWORD *)(v9 + 56),
      *(_QWORD *)(v9 + 64));
    v43 = 16LL * *(unsigned int *)(v9 + 8);
    v44 = *(_QWORD *)v9;
    v45 = v44 + v43;
    while ( v45 != v44 )
    {
      v46 = *(_QWORD *)(v44 + 8);
      v47 = *(_DWORD *)v44;
      v44 += 16;
      sub_B99FD0(v34, v47, v46);
    }
  }
  *(_QWORD *)(a1 + 56) = v34;
  return a1;
}

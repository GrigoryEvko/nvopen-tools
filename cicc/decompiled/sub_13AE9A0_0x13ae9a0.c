// Function: sub_13AE9A0
// Address: 0x13ae9a0
//
__int64 __fastcall sub_13AE9A0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rsi
  __int64 v10; // rdi
  int v11; // edx
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // r10
  __int64 v15; // r8
  __int64 v16; // rsi
  unsigned int v17; // ecx
  __int64 *v18; // rax
  __int64 v19; // r11
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 result; // rax
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // r15
  __int64 v36; // r14
  int v37; // eax
  int v38; // eax
  int v39; // r9d
  int v40; // r8d
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v45; // [rsp+28h] [rbp-118h]
  __int64 v46; // [rsp+28h] [rbp-118h]
  __int64 v47; // [rsp+30h] [rbp-110h]
  int v48; // [rsp+30h] [rbp-110h]
  __int64 v49; // [rsp+38h] [rbp-108h]
  unsigned __int8 v50; // [rsp+38h] [rbp-108h]
  unsigned __int8 v51; // [rsp+38h] [rbp-108h]
  unsigned __int8 v52; // [rsp+38h] [rbp-108h]
  unsigned __int8 v53; // [rsp+38h] [rbp-108h]
  __int64 *v54; // [rsp+48h] [rbp-F8h] BYREF
  unsigned __int64 v55[2]; // [rsp+50h] [rbp-F0h] BYREF
  _BYTE v56[32]; // [rsp+60h] [rbp-E0h] BYREF
  unsigned __int64 v57[2]; // [rsp+80h] [rbp-C0h] BYREF
  _BYTE v58[32]; // [rsp+90h] [rbp-B0h] BYREF
  _BYTE *v59; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v60; // [rsp+B8h] [rbp-88h]
  _BYTE v61[32]; // [rsp+C0h] [rbp-80h] BYREF
  _BYTE *v62; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v63; // [rsp+E8h] [rbp-58h]
  _BYTE v64[80]; // [rsp+F0h] [rbp-50h] BYREF

  v49 = sub_13A4950(a2);
  v47 = sub_13A4950(a3);
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_DWORD *)(v7 + 24);
  if ( v8 )
  {
    v9 = *(_QWORD *)(a2 + 40);
    v10 = *(_QWORD *)(v7 + 8);
    v11 = v8 - 1;
    v12 = v11 & (((unsigned int)*(_QWORD *)(a2 + 40) >> 9) ^ ((unsigned int)v9 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v9 == *v13 )
    {
LABEL_3:
      v15 = v13[1];
    }
    else
    {
      v37 = 1;
      while ( v14 != -8 )
      {
        v40 = v37 + 1;
        v12 = v11 & (v37 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( v9 == *v13 )
          goto LABEL_3;
        v37 = v40;
      }
      v15 = 0;
    }
    v16 = *(_QWORD *)(a3 + 40);
    v17 = v11 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v18 = (__int64 *)(v10 + 16LL * v17);
    v19 = *v18;
    if ( *v18 == v16 )
    {
LABEL_5:
      v20 = v18[1];
    }
    else
    {
      v38 = 1;
      while ( v19 != -8 )
      {
        v39 = v38 + 1;
        v17 = v11 & (v38 + v17);
        v18 = (__int64 *)(v10 + 16LL * v17);
        v19 = *v18;
        if ( *v18 == v16 )
          goto LABEL_5;
        v38 = v39;
      }
      v20 = 0;
    }
  }
  else
  {
    v15 = 0;
    v20 = 0;
  }
  v21 = sub_1472610(*(_QWORD *)(a1 + 8), v49, v15);
  v22 = sub_1472610(*(_QWORD *)(a1 + 8), v47, v20);
  v23 = sub_1456F20(*(_QWORD *)(a1 + 8), v21);
  v24 = *(_QWORD *)(a1 + 8);
  v45 = v23;
  if ( *(_WORD *)(v23 + 24) != 10 )
  {
    sub_1456F20(v24, v22);
    return 0;
  }
  v25 = sub_1456F20(v24, v22);
  if ( *(_WORD *)(v25 + 24) != 10 )
    return 0;
  if ( v45 != v25 )
    return 0;
  v27 = sub_145D1F0(*(_QWORD *)(a1 + 8), a2);
  v28 = sub_145D1F0(*(_QWORD *)(a1 + 8), a3);
  if ( v28 != v27 )
    return 0;
  v29 = sub_14806B0(*(_QWORD *)(a1 + 8), v21, v45, 0, 0);
  v30 = sub_14806B0(*(_QWORD *)(a1 + 8), v22, v45, 0, 0);
  v31 = v30;
  if ( *(_WORD *)(v29 + 24) != 7
    || *(_WORD *)(v30 + 24) != 7
    || *(_QWORD *)(v29 + 40) != 2
    || *(_QWORD *)(v30 + 40) != 2 )
  {
    return 0;
  }
  v32 = *(_QWORD *)(a1 + 8);
  v55[0] = (unsigned __int64)v56;
  v55[1] = 0x400000000LL;
  sub_14857C0(v32, v29, v55);
  sub_14857C0(*(_QWORD *)(a1 + 8), v31, v55);
  v33 = *(_QWORD *)(a1 + 8);
  v57[0] = (unsigned __int64)v58;
  v57[1] = 0x400000000LL;
  sub_14900D0(v33, v55, v57, v28);
  v34 = *(_QWORD *)(a1 + 8);
  v59 = v61;
  v62 = v64;
  v60 = 0x400000000LL;
  v63 = 0x400000000LL;
  sub_14905F0(v34, v29, &v59, v57);
  sub_14905F0(*(_QWORD *)(a1 + 8), v31, &v62, v57);
  if ( (unsigned int)v60 > 1 && (unsigned int)v60 == (unsigned __int64)(unsigned int)v63 && (unsigned int)v63 > 1uLL )
  {
    if ( (int)v60 <= 1 )
    {
      sub_13AE820((__int64)a4, (int)v60);
LABEL_49:
      result = 1;
      goto LABEL_29;
    }
    v35 = 8;
    v46 = 8LL * (unsigned int)v60;
    v36 = v47;
    v48 = v60;
    while ( (unsigned __int8)sub_13A7A70(a1, *(_QWORD *)&v59[v35], v49)
         && (unsigned __int8)sub_13A7900(a1, *(_QWORD *)&v59[v35], *(_QWORD *)(v57[0] + v35 - 8))
         && (unsigned __int8)sub_13A7A70(a1, *(_QWORD *)&v62[v35], v36)
         && (unsigned __int8)sub_13A7900(a1, *(_QWORD *)&v62[v35], *(_QWORD *)(v57[0] + v35 - 8)) )
    {
      v35 += 8;
      if ( v46 == v35 )
      {
        v41 = 0;
        sub_13AE820((__int64)a4, v48);
        do
        {
          v42 = 48 * v41;
          *(_QWORD *)(*a4 + v42) = *(_QWORD *)&v59[8 * v41];
          v43 = *(_QWORD *)&v62[8 * v41++];
          *(_QWORD *)(*a4 + v42 + 8) = v43;
          v54 = (__int64 *)(*a4 + v42);
          sub_13A6D30(a1, &v54, 1);
        }
        while ( v48 > (int)v41 );
        goto LABEL_49;
      }
    }
  }
  result = 0;
LABEL_29:
  if ( v62 != v64 )
  {
    v50 = result;
    _libc_free((unsigned __int64)v62);
    result = v50;
  }
  if ( v59 != v61 )
  {
    v51 = result;
    _libc_free((unsigned __int64)v59);
    result = v51;
  }
  if ( (_BYTE *)v57[0] != v58 )
  {
    v52 = result;
    _libc_free(v57[0]);
    result = v52;
  }
  if ( (_BYTE *)v55[0] != v56 )
  {
    v53 = result;
    _libc_free(v55[0]);
    return v53;
  }
  return result;
}

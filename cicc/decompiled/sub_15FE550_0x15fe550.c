// Function: sub_15FE550
// Address: 0x15fe550
//
__int64 __fastcall sub_15FE550(
        __int64 a1,
        __int64 a2,
        __int64 **a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        double a9,
        __int64 *a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  __int64 v15; // r12
  char v16; // dl
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // r10
  __int64 *v23; // rdx
  int v24; // esi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r10
  __int64 v28; // r14
  __int64 *v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rdx
  int v32; // r8d
  __int64 v33; // rcx
  __int64 v34; // r13
  __int64 v35; // rax
  __int16 v36; // ax
  __int64 v38; // r10
  int v39; // eax
  __int64 *v40; // rax
  __int64 v41; // rbx
  __int64 *v42; // rdx
  int v43; // esi
  __int64 v44; // rax
  __int64 *v45; // rdx
  __int64 v46; // rsi
  __int64 v47; // rax
  int v48; // r8d
  __int64 v49; // rcx
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // [rsp+8h] [rbp-88h]
  int v55; // [rsp+14h] [rbp-7Ch]
  __int64 v56; // [rsp+18h] [rbp-78h]
  char v57; // [rsp+18h] [rbp-78h]
  int v58; // [rsp+20h] [rbp-70h]
  __int64 v59; // [rsp+20h] [rbp-70h]
  __int64 v60; // [rsp+20h] [rbp-70h]
  __int64 v62; // [rsp+28h] [rbp-68h]
  __int64 v64; // [rsp+30h] [rbp-60h]
  __int64 v65; // [rsp+30h] [rbp-60h]
  __int64 v66; // [rsp+30h] [rbp-60h]
  __int64 v67; // [rsp+38h] [rbp-58h] BYREF
  unsigned __int64 v68[2]; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v69[8]; // [rsp+50h] [rbp-40h] BYREF

  v15 = a12;
  v67 = a5;
  if ( a6 )
  {
    if ( a3 != *(__int64 ***)a6 )
    {
      LOWORD(v69[0]) = 257;
      if ( a1 )
        a6 = sub_15FE0A0((_QWORD *)a6, (__int64)a3, 0, (__int64)v68, a1);
      else
        a6 = sub_15FE4E0((_QWORD *)a6, (__int64)a3, 0, (__int64)v68, a2);
    }
  }
  else
  {
    a6 = sub_15A0680((__int64)a3, 1, 0);
  }
  v16 = *(_BYTE *)(a6 + 16);
  if ( v16 == 13 )
  {
    if ( *(_DWORD *)(a6 + 32) <= 0x40u )
    {
      if ( *(_QWORD *)(a6 + 24) == 1 )
        goto LABEL_8;
    }
    else
    {
      v58 = *(_DWORD *)(a6 + 32);
      v56 = a6;
      if ( (unsigned int)sub_16A57B0(a6 + 24) == v58 - 1 )
        goto LABEL_8;
      a6 = v56;
      v16 = 13;
    }
    v38 = v67;
    if ( *(_BYTE *)(v67 + 16) != 13 )
    {
LABEL_31:
      v40 = (__int64 *)sub_15A4750((__int64 ***)a6, a3, 0);
      v67 = sub_15A2C20(v40, v67, 0, 0, a7, a8, a9);
      goto LABEL_8;
    }
  }
  else
  {
    v38 = v67;
    if ( *(_BYTE *)(v67 + 16) != 13 )
      goto LABEL_30;
  }
  if ( *(_DWORD *)(v38 + 32) <= 0x40u )
  {
    if ( *(_QWORD *)(v38 + 24) != 1 )
    {
LABEL_30:
      if ( (unsigned __int8)v16 > 0x10u )
      {
        v68[0] = (unsigned __int64)"mallocsize";
        LOWORD(v69[0]) = 259;
        if ( !a1 )
        {
          v17 = a2;
          v67 = sub_15FB4C0(15, (__int64 *)a6, v38, (__int64)v68, a2);
          goto LABEL_10;
        }
        v67 = sub_15FB440(15, (__int64 *)a6, v38, (__int64)v68, a1);
        goto LABEL_9;
      }
      goto LABEL_31;
    }
  }
  else
  {
    v54 = a6;
    v55 = *(_DWORD *)(v38 + 32);
    v57 = v16;
    v60 = v38;
    v39 = sub_16A57B0(v38 + 24);
    v38 = v60;
    v16 = v57;
    a6 = v54;
    if ( v39 != v55 - 1 )
      goto LABEL_30;
  }
  v67 = a6;
LABEL_8:
  v17 = a2;
  if ( a1 )
LABEL_9:
    v17 = *(_QWORD *)(a1 + 40);
LABEL_10:
  v59 = *(_QWORD *)(*(_QWORD *)(v17 + 56) + 40LL);
  v18 = sub_157E9C0(v17);
  v19 = sub_16471D0(v18, 0);
  if ( !a12 )
  {
    v69[0] = a3;
    v68[0] = (unsigned __int64)v69;
    v68[1] = 0x100000001LL;
    v52 = sub_1644EA0(v19, v69, 1, 0);
    v15 = sub_1632080(v59, "malloc", 6, v52, 0);
    if ( (_QWORD *)v68[0] != v69 )
      _libc_free(v68[0]);
  }
  v62 = sub_1646BA0(a4, 0);
  LOWORD(v69[0]) = 259;
  v20 = &a10[7 * a11];
  v68[0] = (unsigned __int64)"malloccall";
  v21 = *(_QWORD *)v15;
  if ( a1 )
  {
    v22 = *(_QWORD *)(v21 + 24);
    if ( a10 == v20 )
    {
      v66 = *(_QWORD *)(v21 + 24);
      v53 = sub_1648AB0(72, 2, (unsigned int)(16 * a11));
      v27 = v66;
      v28 = v53;
      if ( !v53 )
        goto LABEL_20;
      v33 = -48;
      v32 = 2;
    }
    else
    {
      v23 = a10;
      v24 = 0;
      do
      {
        v25 = v23[5] - v23[4];
        v23 += 7;
        v24 += v25 >> 3;
      }
      while ( v20 != v23 );
      v64 = v22;
      v26 = sub_1648AB0(72, (unsigned int)(v24 + 2), (unsigned int)(16 * a11));
      v27 = v64;
      v28 = v26;
      if ( !v26 )
      {
LABEL_20:
        v34 = v28;
        if ( v62 != *(_QWORD *)v28 )
        {
          v35 = sub_1648A60(56, 1);
          v34 = v35;
          if ( v35 )
            sub_15FD590(v35, v28, v62, a13, a1);
        }
        goto LABEL_23;
      }
      v29 = a10;
      LODWORD(v30) = 0;
      do
      {
        v31 = v29[5] - v29[4];
        v29 += 7;
        v30 = (unsigned int)(v31 >> 3) + (unsigned int)v30;
      }
      while ( &a10[7 * a11] != v29 );
      v32 = v30 + 2;
      v33 = -24 - 8 * (3 * v30 + 3);
    }
    v65 = v27;
    sub_15F1EA0(v28, **(_QWORD **)(v27 + 16), 54, v28 + v33, v32, a1);
    *(_QWORD *)(v28 + 56) = 0;
    sub_15F5B40(v28, v65, v15, &v67, 1, (__int64)v68, a10, a11);
    goto LABEL_20;
  }
  v41 = *(_QWORD *)(v21 + 24);
  if ( a10 == v20 )
  {
    v28 = sub_1648AB0(72, 2, (unsigned int)(16 * a11));
    if ( !v28 )
      goto LABEL_42;
    v49 = -48;
    v48 = 2;
    goto LABEL_41;
  }
  v42 = a10;
  v43 = 0;
  do
  {
    v44 = v42[5] - v42[4];
    v42 += 7;
    v43 += v44 >> 3;
  }
  while ( v20 != v42 );
  v28 = sub_1648AB0(72, (unsigned int)(v43 + 2), (unsigned int)(16 * a11));
  if ( v28 )
  {
    v45 = a10;
    LODWORD(v46) = 0;
    do
    {
      v47 = v45[5] - v45[4];
      v45 += 7;
      v46 = (unsigned int)(v47 >> 3) + (unsigned int)v46;
    }
    while ( v45 != &a10[7 * a11] );
    v48 = v46 + 2;
    v49 = -24 - 8 * (3 * v46 + 3);
LABEL_41:
    sub_15F1EA0(v28, **(_QWORD **)(v41 + 16), 54, v28 + v49, v48, 0);
    *(_QWORD *)(v28 + 56) = 0;
    sub_15F5B40(v28, v41, v15, &v67, 1, (__int64)v68, a10, a11);
  }
LABEL_42:
  v34 = v28;
  if ( v62 != *(_QWORD *)v28 )
  {
    sub_157E9D0(a2 + 40, v28);
    v50 = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(v28 + 32) = a2 + 40;
    v50 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v28 + 24) = v50 | *(_QWORD *)(v28 + 24) & 7LL;
    *(_QWORD *)(v50 + 8) = v28 + 24;
    *(_QWORD *)(a2 + 40) = *(_QWORD *)(a2 + 40) & 7LL | (v28 + 24);
    v51 = sub_1648A60(56, 1);
    v34 = v51;
    if ( v51 )
      sub_15FD590(v51, v28, v62, a13, 0);
  }
LABEL_23:
  v36 = *(_WORD *)(v28 + 18) & 0xFFFC | 1;
  *(_WORD *)(v28 + 18) = v36;
  if ( !*(_BYTE *)(v15 + 16) )
  {
    *(_WORD *)(v28 + 18) = (*(_WORD *)(v15 + 18) >> 2) & 0xFFC | 1 | v36 & 0x8000;
    if ( !(unsigned __int8)sub_1560260((_QWORD *)(v15 + 112), 0, 20) )
      sub_15E0D50(v15, 0, 20);
  }
  return v34;
}

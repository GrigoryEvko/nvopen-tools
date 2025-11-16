// Function: sub_15FD710
// Address: 0x15fd710
//
__int64 __fastcall sub_15FD710(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  _QWORD *v8; // r13
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 *v15; // r15
  __int64 *v16; // rdx
  int v17; // esi
  __int64 v18; // rax
  __int64 v19; // r11
  __int64 *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  int v23; // r8d
  __int64 v24; // rcx
  __int64 v25; // r11
  __int16 v26; // ax
  __int64 v28; // rax
  __int64 v29; // r12
  __int64 *v30; // rcx
  __int64 *v31; // rdx
  int v32; // esi
  __int64 v33; // rax
  __int64 *v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rax
  int v37; // r8d
  __int64 v38; // rcx
  __int64 *v39; // [rsp-10h] [rbp-90h]
  __int64 v40; // [rsp-8h] [rbp-88h]
  __int64 v42; // [rsp+8h] [rbp-78h]
  __int64 v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+10h] [rbp-70h]
  __int64 v45; // [rsp+10h] [rbp-70h]
  __int64 v47; // [rsp+18h] [rbp-68h]
  _QWORD *v48; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v49[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v50[8]; // [rsp+40h] [rbp-40h] BYREF

  v5 = a5;
  if ( a4 )
    v5 = *(_QWORD *)(a4 + 40);
  v8 = *(_QWORD **)(*(_QWORD *)(v5 + 56) + 40LL);
  v42 = sub_1643270(*v8);
  v50[0] = sub_16471D0(*v8, 0);
  v9 = v50[0];
  v49[0] = v50;
  v49[1] = 0x100000001LL;
  v10 = sub_1644EA0(v42, v50, 1, 0);
  v11 = sub_1632080(v8, "free", 4, v10, 0);
  v48 = a1;
  v12 = *a1;
  if ( !a4 )
  {
    if ( v9 != v12 )
    {
      LOWORD(v50[0]) = 257;
      v28 = sub_1648A60(56, 1);
      v29 = v28;
      if ( v28 )
        sub_15FD650(v28, (__int64)a1, v9, (__int64)v49, a5);
      v48 = (_QWORD *)v29;
    }
    LOWORD(v50[0]) = 257;
    v14 = *(_QWORD *)(*(_QWORD *)v11 + 24LL);
    v30 = &a2[7 * a3];
    if ( v30 == a2 )
    {
      v19 = sub_1648AB0(72, 2, (unsigned int)(16 * a3));
      if ( !v19 )
        goto LABEL_17;
      v38 = -48;
      v37 = 2;
    }
    else
    {
      v31 = a2;
      v32 = 0;
      do
      {
        v33 = v31[5] - v31[4];
        v31 += 7;
        v32 += v33 >> 3;
      }
      while ( v30 != v31 );
      v19 = sub_1648AB0(72, (unsigned int)(v32 + 2), (unsigned int)(16 * a3));
      if ( !v19 )
        goto LABEL_17;
      v34 = a2;
      LODWORD(v35) = 0;
      do
      {
        v36 = v34[5] - v34[4];
        v34 += 7;
        v35 = (unsigned int)(v36 >> 3) + (unsigned int)v35;
      }
      while ( &a2[7 * a3] != v34 );
      v37 = v35 + 2;
      v38 = -24 - 8 * (3 * v35 + 3);
    }
    v45 = v19;
    sub_15F1EA0(v19, **(_QWORD **)(v14 + 16), 54, v19 + v38, v37, 0);
    v25 = v45;
    *(_QWORD *)(v45 + 56) = 0;
    v40 = a3;
    v39 = a2;
    goto LABEL_16;
  }
  if ( v9 != v12 )
  {
    LOWORD(v50[0]) = 257;
    v13 = sub_1648A60(56, 1);
    if ( v13 )
    {
      v43 = v13;
      sub_15FD590(v13, (__int64)a1, v9, (__int64)v49, a4);
      v13 = v43;
    }
    v48 = (_QWORD *)v13;
  }
  LOWORD(v50[0]) = 257;
  v14 = *(_QWORD *)(*(_QWORD *)v11 + 24LL);
  v15 = &a2[7 * a3];
  if ( v15 == a2 )
  {
    v19 = sub_1648AB0(72, 2, (unsigned int)(16 * a3));
    if ( v19 )
    {
      v24 = -48;
      v23 = 2;
LABEL_15:
      v44 = v19;
      sub_15F1EA0(v19, **(_QWORD **)(v14 + 16), 54, v19 + v24, v23, a4);
      v25 = v44;
      *(_QWORD *)(v44 + 56) = 0;
      v40 = a3;
      v39 = a2;
LABEL_16:
      v47 = v25;
      sub_15F5B40(v25, v14, v11, (__int64 *)&v48, 1, (__int64)v49, v39, v40);
      v19 = v47;
    }
  }
  else
  {
    v16 = a2;
    v17 = 0;
    do
    {
      v18 = v16[5] - v16[4];
      v16 += 7;
      v17 += v18 >> 3;
    }
    while ( v15 != v16 );
    v19 = sub_1648AB0(72, (unsigned int)(v17 + 2), (unsigned int)(16 * a3));
    if ( v19 )
    {
      v20 = a2;
      LODWORD(v21) = 0;
      do
      {
        v22 = v20[5] - v20[4];
        v20 += 7;
        v21 = (unsigned int)(v22 >> 3) + (unsigned int)v21;
      }
      while ( v15 != v20 );
      v23 = v21 + 2;
      v24 = -24 - 8 * (3 * v21 + 3);
      goto LABEL_15;
    }
  }
LABEL_17:
  v26 = *(_WORD *)(v19 + 18) & 0xFFFC | 1;
  *(_WORD *)(v19 + 18) = v26;
  if ( !*(_BYTE *)(v11 + 16) )
    *(_WORD *)(v19 + 18) = (*(_WORD *)(v11 + 18) >> 2) & 0xFFC | 1 | v26 & 0x8000;
  return v19;
}

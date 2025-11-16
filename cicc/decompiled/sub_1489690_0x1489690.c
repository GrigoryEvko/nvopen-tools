// Function: sub_1489690
// Address: 0x1489690
//
__int64 __fastcall sub_1489690(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        unsigned int a9)
{
  char v9; // al
  unsigned int v10; // r9d
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v14; // rdx
  bool v15; // zf
  __int64 v16; // r14
  __int64 v17; // rbx
  __int16 v18; // cx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // rcx
  char v29; // si
  __int64 v30; // r14
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // r14
  __int64 v35; // rdx
  __int64 v36; // r14
  __int64 v37; // rbx
  __int64 v38; // rax
  char v39; // al
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r14
  char v44; // al
  unsigned __int8 v45; // [rsp+7h] [rbp-79h]
  __int64 v46; // [rsp+8h] [rbp-78h]
  __int64 v47; // [rsp+8h] [rbp-78h]
  unsigned __int8 v48; // [rsp+8h] [rbp-78h]
  __int64 v49; // [rsp+10h] [rbp-70h]
  __int64 v50; // [rsp+10h] [rbp-70h]
  unsigned __int8 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+18h] [rbp-68h] BYREF
  __int64 v54; // [rsp+28h] [rbp-58h] BYREF
  __int64 v55; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v56; // [rsp+38h] [rbp-48h]
  __int64 *v57; // [rsp+40h] [rbp-40h]
  unsigned int *v58; // [rsp+48h] [rbp-38h]

  v53 = a6;
  v9 = a9;
  v10 = 0;
  if ( a9 <= dword_4F9AFC0 )
  {
    v11 = a3;
    v12 = a4;
    if ( a2 == 40 )
    {
      v14 = v53;
      v53 = a5;
    }
    else
    {
      if ( a2 != 38 )
        return v10;
      v12 = a3;
      v14 = a5;
      v11 = a4;
    }
    v15 = *(_WORD *)(v12 + 24) == 3;
    v54 = v14;
    v16 = v12;
    if ( v15 )
      v16 = *(_QWORD *)(v12 + 32);
    v17 = v14;
    if ( *(_WORD *)(v14 + 24) == 3 )
      v17 = *(_QWORD *)(v14 + 32);
    v55 = a1;
    v56 = &v54;
    v57 = &v53;
    v58 = &a9;
    v18 = *(_WORD *)(v16 + 24);
    if ( v18 == 4 )
    {
      v19 = sub_1456040(v16);
      v20 = sub_1456C90(a1, v19);
      v21 = sub_1456040(v11);
      v22 = sub_1456C90(a1, v21);
      v10 = 0;
      if ( v20 == v22 && (*(_BYTE *)(v16 + 26) & 4) != 0 )
      {
        v23 = *(__int64 **)(v16 + 32);
        v24 = *v23;
        v49 = v23[1];
        v25 = sub_1456040(v11);
        v26 = sub_145CF80(a1, v25, 1, 0);
        v27 = sub_1480620(a1, v26, 0);
        if ( ((unsigned __int8)sub_1481140(v55, 0x26u, v24, v27)
           || (unsigned __int8)sub_1489690(v55, 38, v24, v27, *v56, *v57, *v58 + 1))
          && ((unsigned __int8)sub_1481140(v55, 0x26u, v49, v11)
           || (unsigned __int8)sub_1489690(v55, 38, v49, v11, *v56, *v57, *v58 + 1))
          || ((unsigned __int8)sub_1481140(v55, 0x26u, v49, v27)
           || (unsigned __int8)sub_1489690(v55, 38, v49, v27, *v56, *v57, *v58 + 1))
          && (unsigned __int8)sub_1489BA0(&v55, v24, v11) )
        {
          return 1;
        }
        goto LABEL_28;
      }
    }
    else
    {
      if ( v18 != 10 )
        return (unsigned int)sub_1488C70(a1, 0x26u, v12, v11, v14, v53, a7, a8, v9 + 1);
      v28 = *(_QWORD *)(v16 - 8);
      v29 = *(_BYTE *)(v28 + 16);
      if ( v29 == 42 )
      {
        v30 = *(_QWORD *)(v28 - 48);
        if ( !v30 )
          return (unsigned int)sub_1488C70(a1, 0x26u, v12, v11, v14, v53, a7, a8, v9 + 1);
        v31 = *(_QWORD *)(v28 - 24);
        if ( !v31 )
          return (unsigned int)sub_1488C70(a1, 0x26u, v12, v11, v14, v53, a7, a8, v9 + 1);
      }
      else
      {
        if ( v29 != 5 )
          return (unsigned int)sub_1488C70(a1, 0x26u, v12, v11, v14, v53, a7, a8, v9 + 1);
        if ( *(_WORD *)(v28 + 18) != 18 )
          return (unsigned int)sub_1488C70(a1, 0x26u, v12, v11, v14, v53, a7, a8, v9 + 1);
        v30 = *(_QWORD *)(v28 - 24LL * (*(_DWORD *)(v28 + 20) & 0xFFFFFFF));
        if ( !v30 )
          return (unsigned int)sub_1488C70(a1, 0x26u, v12, v11, v14, v53, a7, a8, v9 + 1);
        v31 = *(_QWORD *)(v28 + 24 * (1LL - (*(_DWORD *)(v28 + 20) & 0xFFFFFFF)));
        if ( !v31 )
          return (unsigned int)sub_1488C70(a1, 0x26u, v12, v11, v14, v53, a7, a8, v9 + 1);
      }
      if ( *(_BYTE *)(v31 + 16) != 13 )
        return 0;
      v46 = sub_146F1B0(a1, v31);
      v32 = sub_14646A0(a1, v30);
      v33 = v32;
      if ( !v32 )
        return 0;
      v50 = sub_1456040(v32);
      if ( v50 != sub_1456040(v17) )
        return 0;
      if ( !(unsigned __int8)sub_1452FA0(v33, v17) )
        return 0;
      v51 = sub_1477C30(a1, v46);
      if ( !v51 )
        return 0;
      v34 = **(_QWORD **)(v46 + 32);
      v35 = sub_1456040(v53);
      if ( (*(_BYTE *)(v35 + 8) == 15) != (*(_BYTE *)(v34 + 8) == 15) )
        return 0;
      v45 = v51;
      v36 = sub_1456E50(a1, v34, v35);
      v52 = sub_147BE00(a1, v46, v36);
      v37 = sub_147BE00(a1, v53, v36);
      v38 = sub_145CF80(a1, v36, 2, 0);
      v47 = sub_14806B0(a1, v52, v38, 0, 0);
      v39 = sub_1477A90(a1, v11);
      LOBYTE(v10) = v45;
      if ( !v39 || (v40 = sub_1489BA0(&v55, v37, v47), v10 = v45, !v40) )
      {
        v48 = v10;
        v41 = sub_145CF80(a1, v36, 1, 0);
        v42 = sub_1480620(a1, v41, 0);
        v43 = sub_14806B0(a1, v42, v52, 0, 0);
        if ( !(unsigned __int8)sub_1477B50(a1, v11) || (v44 = sub_1489BA0(&v55, v37, v43), v10 = v48, !v44) )
        {
LABEL_28:
          v9 = a9;
          v14 = v54;
          return (unsigned int)sub_1488C70(a1, 0x26u, v12, v11, v14, v53, a7, a8, v9 + 1);
        }
      }
    }
  }
  return v10;
}

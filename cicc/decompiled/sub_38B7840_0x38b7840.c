// Function: sub_38B7840
// Address: 0x38b7840
//
__int64 __fastcall sub_38B7840(__int64 a1, double a2, double a3, double a4)
{
  __int16 *v4; // rbx
  unsigned int v5; // r12d
  void *v6; // rbx
  __int64 v8; // rdx
  __int64 v9; // r13
  const char *v10; // rax
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // r14
  __int64 v15; // rsi
  __int64 v16; // r13
  const char *v17; // rax
  __int64 v18; // r13
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned __int64 v22; // [rsp+0h] [rbp-220h]
  __int64 v23; // [rsp+8h] [rbp-218h]
  _QWORD v24[2]; // [rsp+40h] [rbp-1E0h] BYREF
  char v25; // [rsp+50h] [rbp-1D0h]
  char v26; // [rsp+51h] [rbp-1CFh]
  _BYTE *v27; // [rsp+60h] [rbp-1C0h] BYREF
  __int64 v28; // [rsp+68h] [rbp-1B8h]
  _BYTE v29[64]; // [rsp+70h] [rbp-1B0h] BYREF
  int v30; // [rsp+B0h] [rbp-170h] BYREF
  unsigned __int64 v31; // [rsp+B8h] [rbp-168h]
  unsigned int v32; // [rsp+C0h] [rbp-160h]
  __int64 v33; // [rsp+C8h] [rbp-158h]
  _QWORD *v34; // [rsp+D0h] [rbp-150h]
  __int64 v35; // [rsp+D8h] [rbp-148h]
  _BYTE v36[16]; // [rsp+E0h] [rbp-140h] BYREF
  _QWORD *v37; // [rsp+F0h] [rbp-130h]
  __int64 v38; // [rsp+F8h] [rbp-128h]
  _BYTE v39[16]; // [rsp+100h] [rbp-120h] BYREF
  unsigned __int64 v40; // [rsp+110h] [rbp-110h]
  unsigned int v41; // [rsp+118h] [rbp-108h]
  char v42; // [rsp+11Ch] [rbp-104h]
  void *v43; // [rsp+128h] [rbp-F8h] BYREF
  __int64 v44; // [rsp+130h] [rbp-F0h]
  unsigned __int64 v45; // [rsp+148h] [rbp-D8h]
  __int64 v46; // [rsp+150h] [rbp-D0h] BYREF
  unsigned __int64 v47; // [rsp+158h] [rbp-C8h]
  __int64 v48; // [rsp+168h] [rbp-B8h]
  unsigned __int8 *v49; // [rsp+170h] [rbp-B0h]
  size_t v50; // [rsp+178h] [rbp-A8h]
  _BYTE v51[16]; // [rsp+180h] [rbp-A0h] BYREF
  _QWORD *v52; // [rsp+190h] [rbp-90h]
  __int64 v53; // [rsp+198h] [rbp-88h]
  _BYTE v54[16]; // [rsp+1A0h] [rbp-80h] BYREF
  unsigned __int64 v55; // [rsp+1B0h] [rbp-70h]
  unsigned int v56; // [rsp+1B8h] [rbp-68h]
  char v57; // [rsp+1BCh] [rbp-64h]
  void *v58; // [rsp+1C8h] [rbp-58h] BYREF
  __int64 v59; // [rsp+1D0h] [rbp-50h]
  unsigned __int64 v60; // [rsp+1E8h] [rbp-38h]

  v22 = *(_QWORD *)(a1 + 56);
  v23 = a1 + 8;
  v30 = 0;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v34 = v36;
  v31 = 0;
  v33 = 0;
  v35 = 0;
  v36[0] = 0;
  v37 = v39;
  v38 = 0;
  v39[0] = 0;
  v41 = 1;
  v40 = 0;
  v42 = 0;
  v4 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)&v46, 0.0);
  sub_169E320(&v43, &v46, v4);
  sub_1698460((__int64)&v46);
  v54[0] = 0;
  v57 = 0;
  v49 = v51;
  v52 = v54;
  v45 = 0;
  LODWORD(v46) = 0;
  v47 = 0;
  v48 = 0;
  v50 = 0;
  v51[0] = 0;
  v53 = 0;
  v56 = 1;
  v55 = 0;
  sub_169D3F0((__int64)&v27, 0.0);
  sub_169E320(&v58, (__int64 *)&v27, v4);
  sub_1698460((__int64)&v27);
  v60 = 0;
  v27 = v29;
  v28 = 0x1000000000LL;
  if ( (unsigned __int8)sub_389C540(a1, (__int64)&v30, 0.0, a3, a4)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected comma in uselistorder_bb directive")
    || (unsigned __int8)sub_389C540(a1, (__int64)&v46, 0.0, a3, a4)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected comma in uselistorder_bb directive")
    || (unsigned __int8)sub_388EBB0(a1, (__int64)&v27) )
  {
    v5 = 1;
    goto LABEL_3;
  }
  if ( v30 == 3 )
  {
    v9 = sub_1632000(*(_QWORD *)(a1 + 176), (__int64)v34, v35);
  }
  else
  {
    if ( v30 != 1 )
    {
LABEL_51:
      v26 = 1;
      v17 = "expected function name in uselistorder_bb";
LABEL_52:
      v24[0] = v17;
      v25 = 3;
      v5 = sub_38814C0(v23, v31, (__int64)v24);
      goto LABEL_3;
    }
    v8 = *(_QWORD *)(a1 + 1000);
    if ( v32 >= (unsigned __int64)((*(_QWORD *)(a1 + 1008) - v8) >> 3) )
    {
LABEL_53:
      v26 = 1;
      v17 = "invalid function forward reference in uselistorder_bb";
      goto LABEL_52;
    }
    v9 = *(_QWORD *)(v8 + 8LL * v32);
  }
  if ( !v9 )
    goto LABEL_53;
  if ( *(_BYTE *)(v9 + 16) )
    goto LABEL_51;
  if ( sub_15E4F60(v9) )
  {
    v26 = 1;
    v17 = "invalid declaration in uselistorder_bb";
    goto LABEL_52;
  }
  if ( !(_DWORD)v46 )
  {
    v26 = 1;
    v10 = "invalid numeric label in uselistorder_bb";
    goto LABEL_42;
  }
  if ( (_DWORD)v46 != 2 )
  {
    v26 = 1;
    v10 = "expected basic block name in uselistorder_bb";
LABEL_42:
    v24[0] = v10;
    v25 = 3;
    v5 = sub_38814C0(v23, v47, (__int64)v24);
    goto LABEL_3;
  }
  v18 = *(_QWORD *)(v9 + 104);
  v19 = sub_16D1B30((__int64 *)v18, v49, v50);
  if ( v19 == -1
    || (v20 = *(_QWORD *)v18 + 8LL * v19, v20 == *(_QWORD *)v18 + 8LL * *(unsigned int *)(v18 + 8))
    || (v21 = *(_QWORD *)(*(_QWORD *)v20 + 8LL)) == 0 )
  {
    v26 = 1;
    v10 = "invalid basic block in uselistorder_bb";
    goto LABEL_42;
  }
  if ( *(_BYTE *)(v21 + 16) != 18 )
  {
    v26 = 1;
    v10 = "expected basic block in uselistorder_bb";
    goto LABEL_42;
  }
  v5 = sub_38B6F20(a1, v21, (__int64)v27, (unsigned int)v28, v22);
LABEL_3:
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  if ( v60 )
    j_j___libc_free_0_0(v60);
  v6 = sub_16982C0();
  if ( v58 == v6 )
  {
    v14 = v59;
    if ( v59 )
    {
      v15 = 32LL * *(_QWORD *)(v59 - 8);
      v16 = v59 + v15;
      if ( v59 != v59 + v15 )
      {
        do
        {
          v16 -= 32;
          sub_127D120((_QWORD *)(v16 + 8));
        }
        while ( v14 != v16 );
      }
      j_j_j___libc_free_0_0(v14 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v58);
  }
  if ( v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  if ( v52 != (_QWORD *)v54 )
    j_j___libc_free_0((unsigned __int64)v52);
  if ( v49 != v51 )
    j_j___libc_free_0((unsigned __int64)v49);
  if ( v45 )
    j_j___libc_free_0_0(v45);
  if ( v6 == v43 )
  {
    v11 = v44;
    if ( v44 )
    {
      v12 = 32LL * *(_QWORD *)(v44 - 8);
      v13 = v44 + v12;
      if ( v44 != v44 + v12 )
      {
        do
        {
          v13 -= 32;
          sub_127D120((_QWORD *)(v13 + 8));
        }
        while ( v11 != v13 );
      }
      j_j_j___libc_free_0_0(v11 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v43);
  }
  if ( v41 > 0x40 && v40 )
    j_j___libc_free_0_0(v40);
  if ( v37 != (_QWORD *)v39 )
    j_j___libc_free_0((unsigned __int64)v37);
  if ( v34 != (_QWORD *)v36 )
    j_j___libc_free_0((unsigned __int64)v34);
  return v5;
}

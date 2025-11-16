// Function: sub_258BA20
// Address: 0x258ba20
//
void __fastcall sub_258BA20(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        unsigned __int64 a4,
        unsigned __int8 *a5,
        char a6,
        __int64 a7)
{
  _BYTE *v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // r14
  __int64 v16; // rdx
  char v17; // r12
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  bool v23; // zf
  char v24; // al
  __int64 (__fastcall *v25)(__int64); // rax
  char v26; // al
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int8 *v31; // r14
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rcx
  __int64 (__fastcall *v36)(__int64); // rax
  char v37; // al
  _QWORD *v38; // rax
  _BYTE *v39; // rax
  __int64 v40; // r8
  unsigned __int8 *v41; // rdx
  __int64 v42; // r15
  __int64 i; // r14
  __int64 v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  __int64 v55; // [rsp+8h] [rbp-88h]
  unsigned __int64 v56; // [rsp+8h] [rbp-88h]
  __int64 v57; // [rsp+10h] [rbp-80h]
  int v58; // [rsp+10h] [rbp-80h]
  __int64 v59; // [rsp+10h] [rbp-80h]
  __int64 v61; // [rsp+20h] [rbp-70h]
  __int64 v62; // [rsp+20h] [rbp-70h]
  __int64 v64; // [rsp+30h] [rbp-60h] BYREF
  __int64 v65; // [rsp+38h] [rbp-58h]
  _BYTE *v66; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int8 *v67; // [rsp+48h] [rbp-48h]
  char v68; // [rsp+50h] [rbp-40h]

  v10 = (_BYTE *)a4;
  v11 = 0;
  v64 = sub_250D2C0(a4, 0);
  v65 = v12;
  if ( !a5 )
    goto LABEL_9;
  v13 = *a5;
  if ( (unsigned __int8)(v13 - 34) > 0x33u )
    goto LABEL_9;
  v14 = 0x8000000000041LL;
  if ( !_bittest64(&v14, (unsigned int)(v13 - 34)) )
    goto LABEL_9;
  if ( v13 == 40 )
  {
    v15 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a5);
  }
  else
  {
    v15 = -32;
    if ( v13 != 85 )
    {
      v15 = -96;
      if ( v13 != 34 )
        BUG();
    }
  }
  if ( (a5[7] & 0x80u) != 0 )
  {
    v27 = sub_BD2BC0((__int64)a5);
    v57 = v28 + v27;
    if ( (a5[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v57 >> 4) )
        goto LABEL_26;
    }
    else
    {
      if ( !(unsigned int)((v57 - sub_BD2BC0((__int64)a5)) >> 4) )
        goto LABEL_26;
      if ( (a5[7] & 0x80u) != 0 )
      {
        v58 = *(_DWORD *)(sub_BD2BC0((__int64)a5) + 8);
        if ( (a5[7] & 0x80u) == 0 )
          BUG();
        v29 = sub_BD2BC0((__int64)a5);
        v15 -= 32LL * (unsigned int)(*(_DWORD *)(v29 + v30 - 4) - v58);
        goto LABEL_26;
      }
    }
    BUG();
  }
LABEL_26:
  v31 = &a5[v15];
  v32 = 32LL * (*((_DWORD *)a5 + 1) & 0x7FFFFFF);
  if ( &a5[-v32] != v31 )
  {
    v11 = (__int64)&a5[-v32];
    while ( v10 != *(_BYTE **)v11 )
    {
      v11 += 32;
      if ( v31 == (unsigned __int8 *)v11 )
        goto LABEL_9;
    }
    v11 = (v11 - (__int64)&a5[-v32]) >> 5;
    v64 = sub_254C9B0((__int64)a5, v11);
    v65 = v16;
  }
LABEL_9:
  if ( *(_BYTE *)(sub_250D180(&v64, v11) + 8) != 12 )
  {
LABEL_10:
    if ( *v10 == 17 )
      a5 = 0;
    v17 = a6 | 2;
    v18 = sub_250C180((__int64)v10, a7);
    v66 = v10;
    v23 = v18 == 0;
    v24 = a6;
    v67 = a5;
    if ( v23 )
      v24 = v17;
    v68 = v24;
    v25 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 16LL);
    if ( v25 == sub_2505E60 )
      v26 = a3[17];
    else
      v26 = v25((__int64)a3);
    if ( v26 )
      sub_2579B00((__int64)a3, (__int64)&v66, v19, v20, v21, v22);
    return;
  }
  v59 = sub_250D180((__int64 *)(a1 + 72), v11);
  if ( *(_BYTE *)sub_250D070(&v64) > 0x15u )
  {
    if ( *(_BYTE *)(v59 + 8) == 12 )
    {
      v38 = (_QWORD *)sub_2589400(a2, v64, v65, a1, 2, 0, 1);
      if ( v38 )
      {
        v55 = (__int64)v38;
        v39 = sub_25533A0(v38, a2, 0);
        v40 = v55;
        v67 = v41;
        v66 = v39;
        if ( !(_BYTE)v41 )
        {
          sub_250ED80(a2, v55, a1, 1);
          return;
        }
        v56 = (unsigned __int64)v66;
        if ( v66 )
        {
          sub_250ED80(a2, v40, a1, 1);
          v33 = sub_250C3F0(v56, v59);
          if ( v33 )
            goto LABEL_33;
        }
      }
    }
  }
  else
  {
    v33 = sub_250D070(&v64);
    if ( v33 )
    {
LABEL_33:
      v10 = (_BYTE *)v33;
      goto LABEL_10;
    }
  }
  v34 = v64;
  v35 = sub_25803A0(a2, v64, v65, a1, 1, 0, 1);
  if ( !v35
    || ((v36 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)(v35 + 88) + 16LL), v36 != sub_2505E40)
      ? (v62 = v35, v37 = ((__int64 (__fastcall *)(__int64, __int64))v36)(v35 + 88, v34), v35 = v62)
      : (v37 = *(_BYTE *)(v35 + 105)),
        !v37) )
  {
    v33 = (unsigned __int64)v10;
    goto LABEL_33;
  }
  v42 = *(_QWORD *)(v35 + 144);
  for ( i = v42 + 16LL * *(unsigned int *)(v35 + 152); i != v42; v35 = v61 )
  {
    v44 = v42;
    v61 = v35;
    v42 += 16;
    v45 = sub_AD8D80(v59, v44);
    v67 = 0;
    v66 = (_BYTE *)v45;
    v68 = a6;
    sub_2579EE0(a3, (__int64)&v66, v46, v47, v48, v49);
  }
  if ( *(_BYTE *)(v35 + 288) )
  {
    v50 = sub_ACA8A0((__int64 **)v59);
    v68 = a6;
    v66 = (_BYTE *)v50;
    v67 = 0;
    sub_2579EE0(a3, (__int64)&v66, v51, v52, v53, v54);
  }
}

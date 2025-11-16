// Function: sub_2816B50
// Address: 0x2816b50
//
__int64 __fastcall sub_2816B50(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, _BYTE *a5, char a6, int a7)
{
  __int64 v9; // r9
  char v10; // al
  __int64 v11; // r13
  char *v12; // r15
  __int64 *v13; // rax
  __int64 *v14; // r13
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 result; // rax
  __int64 v20; // r9
  char v21; // cl
  __int64 v22; // r13
  char *v23; // r14
  __int64 *v24; // rax
  __int64 v25; // r9
  __int64 *v26; // r13
  __int64 v27; // rax
  __int64 *v28; // rax
  _BYTE *v29; // r14
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  void (__fastcall *v32)(unsigned __int64); // rax
  __int64 v33; // [rsp+0h] [rbp-F0h]
  __int64 v34; // [rsp+8h] [rbp-E8h]
  __int64 *v35; // [rsp+10h] [rbp-E0h]
  __int64 v36; // [rsp+10h] [rbp-E0h]
  _BYTE *v37; // [rsp+10h] [rbp-E0h]
  _BYTE *v38; // [rsp+18h] [rbp-D8h]
  __int64 *v39; // [rsp+18h] [rbp-D8h]
  unsigned __int8 v40; // [rsp+18h] [rbp-D8h]
  unsigned __int8 v41; // [rsp+18h] [rbp-D8h]
  __int64 v42; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE v43[8]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v44; // [rsp+38h] [rbp-B8h]
  __int64 *v45; // [rsp+40h] [rbp-B0h]
  unsigned __int64 v46; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v47; // [rsp+58h] [rbp-98h]
  __int64 v48; // [rsp+60h] [rbp-90h]
  __int64 v49; // [rsp+68h] [rbp-88h] BYREF
  unsigned int v50; // [rsp+70h] [rbp-80h]
  __int16 v51; // [rsp+A8h] [rbp-48h] BYREF
  __int64 v52; // [rsp+B0h] [rbp-40h]
  char *v53; // [rsp+B8h] [rbp-38h]

  if ( a7 == 1 )
  {
    sub_2297CA0((__int64 *)&v46, *(_QWORD *)(a1 + 1232), (__int64)a4, a5);
    v31 = v46;
    if ( !v46 )
      return 1;
    goto LABEL_35;
  }
  if ( a7 == 2 )
  {
    v9 = 0;
    if ( *a4 > 0x1Cu && (unsigned __int8)(*a4 - 61) <= 1u )
      v9 = *((_QWORD *)a4 - 4);
    v10 = *a5;
    if ( *a5 <= 0x1Cu )
      goto LABEL_38;
    if ( v10 == 61 )
    {
      v11 = *((_QWORD *)a5 - 4);
      if ( !v11 )
        goto LABEL_38;
    }
    else
    {
      if ( v10 != 62 )
        goto LABEL_38;
      v11 = *((_QWORD *)a5 - 4);
      if ( !v11 )
        goto LABEL_38;
    }
    if ( !v9 )
      goto LABEL_38;
    v12 = *(char **)(a3 + 40);
    v33 = (__int64)a5;
    v34 = *(_QWORD *)(a2 + 40);
    v35 = sub_DDFBA0(*(_QWORD *)(a1 + 1240), v9, (char *)v34);
    v13 = sub_DDFBA0(*(_QWORD *)(a1 + 1240), v11, v12);
    v47 = 0;
    v14 = v13;
    v15 = *(_QWORD *)(a1 + 1240);
    v48 = 1;
    v46 = v15;
    v16 = &v49;
    do
    {
      *v16 = -4096;
      v16 += 2;
    }
    while ( v16 != (__int64 *)&v51 );
    v52 = v34;
    v51 = 257;
    v53 = v12;
    v17 = sub_2815B30((__int64)&v46, v35, (__int64)&v51, v34, v33, (__int64)v35);
    a5 = (_BYTE *)v33;
    v38 = (_BYTE *)v17;
    if ( !(_BYTE)v51
      || (v18 = **(_QWORD **)(v34 + 32),
          v43[0] = 0,
          v44 = a1,
          v42 = v18,
          v45 = &v42,
          sub_28149F0((__int64)v14, (__int64)v43),
          a5 = (_BYTE *)v33,
          v43[0]) )
    {
      if ( (v48 & 1) != 0 )
        goto LABEL_38;
      LOBYTE(result) = 0;
    }
    else
    {
      result = sub_DC3A60(*(_QWORD *)(a1 + 1240), (a6 == 0) + 38LL, v38, v14);
      a5 = (_BYTE *)v33;
      if ( (v48 & 1) != 0 )
        goto LABEL_16;
    }
    v37 = a5;
    v41 = result;
    sub_C7D6A0(v49, 16LL * v50, 8);
    a5 = v37;
    result = v41;
LABEL_16:
    if ( (_BYTE)result )
      return result;
LABEL_38:
    sub_2297CA0((__int64 *)&v46, *(_QWORD *)(a1 + 1232), (__int64)a4, a5);
    v31 = v46;
    if ( !v46 )
      return 1;
LABEL_35:
    v32 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v31 + 8LL);
    if ( v32 == sub_228A6E0 )
    {
      j_j___libc_free_0(v31);
      return 0;
    }
    ((void (*)(void))v32)();
    return 0;
  }
  if ( a7 )
    BUG();
  v20 = 0;
  if ( *a4 > 0x1Cu && (unsigned __int8)(*a4 - 61) <= 1u )
    v20 = *((_QWORD *)a4 - 4);
  v21 = *a5;
  result = 0;
  if ( *a5 > 0x1Cu && (v21 == 61 || v21 == 62) )
  {
    v22 = *((_QWORD *)a5 - 4);
    if ( v22 )
    {
      if ( !v20 )
        return 0;
      v23 = *(char **)(a3 + 40);
      v36 = *(_QWORD *)(a2 + 40);
      v39 = sub_DDFBA0(*(_QWORD *)(a1 + 1240), v20, (char *)v36);
      v24 = sub_DDFBA0(*(_QWORD *)(a1 + 1240), v22, v23);
      v47 = 0;
      v26 = v24;
      v27 = *(_QWORD *)(a1 + 1240);
      v48 = 1;
      v46 = v27;
      v28 = &v49;
      do
      {
        *v28 = -4096;
        v28 += 2;
      }
      while ( v28 != (__int64 *)&v51 );
      v52 = v36;
      v53 = v23;
      v51 = 257;
      v29 = (_BYTE *)sub_2815B30((__int64)&v46, v39, 257, v36, (__int64)v39, v25);
      if ( !(_BYTE)v51
        || (v30 = **(_QWORD **)(v36 + 32),
            v43[0] = 0,
            v44 = a1,
            v42 = v30,
            v45 = &v42,
            sub_28149F0((__int64)v26, (__int64)v43),
            v43[0]) )
      {
        result = 0;
      }
      else
      {
        result = sub_DC3A60(*(_QWORD *)(a1 + 1240), (a6 == 0) + 38LL, v29, v26);
      }
      if ( (v48 & 1) == 0 )
      {
        v40 = result;
        sub_C7D6A0(v49, 16LL * v50, 8);
        return v40;
      }
    }
  }
  return result;
}

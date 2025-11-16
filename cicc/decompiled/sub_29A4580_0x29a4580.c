// Function: sub_29A4580
// Address: 0x29a4580
//
__int64 __fastcall sub_29A4580(unsigned __int8 *a1)
{
  __int64 v1; // rax
  _BYTE *v2; // rdx
  unsigned int v3; // r8d
  __int64 v5; // r15
  __int64 v6; // rbx
  unsigned __int8 *v7; // rax
  __int64 v8; // rbx
  unsigned int v9; // eax
  unsigned __int8 *v10; // r10
  unsigned __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // eax
  unsigned __int8 *v16; // r10
  unsigned __int8 *v17; // r13
  unsigned __int8 v18; // r8
  int v19; // eax
  unsigned __int64 v20; // rsi
  unsigned __int8 *v21; // rax
  unsigned __int8 *v22; // r13
  unsigned __int8 v23; // r8
  _QWORD *v24; // [rsp+0h] [rbp-A0h]
  unsigned __int64 *v25; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v26; // [rsp+8h] [rbp-98h]
  __int64 v27; // [rsp+8h] [rbp-98h]
  unsigned __int8 v28; // [rsp+8h] [rbp-98h]
  unsigned int v29; // [rsp+8h] [rbp-98h]
  unsigned __int8 *v30; // [rsp+8h] [rbp-98h]
  unsigned __int8 v31; // [rsp+8h] [rbp-98h]
  unsigned int v32; // [rsp+8h] [rbp-98h]
  unsigned __int8 v33; // [rsp+8h] [rbp-98h]
  unsigned __int8 v34; // [rsp+8h] [rbp-98h]
  unsigned __int64 v35; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v36; // [rsp+18h] [rbp-88h]
  _QWORD *v37; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v38; // [rsp+28h] [rbp-78h]
  __int64 v39; // [rsp+30h] [rbp-70h] BYREF
  __int16 v40; // [rsp+38h] [rbp-68h]
  unsigned __int64 *v41; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v42; // [rsp+48h] [rbp-58h]
  unsigned __int64 *v43; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v44; // [rsp+58h] [rbp-48h]
  unsigned __int64 *v45; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v46; // [rsp+68h] [rbp-38h]

  v1 = sub_B491C0((__int64)a1);
  v2 = (_BYTE *)*((_QWORD *)a1 - 4);
  v3 = 0;
  if ( *v2 != 61 )
    return v3;
  v5 = *(_QWORD *)(v1 + 40);
  v6 = *((_QWORD *)v2 - 4);
  v36 = sub_AE43F0(v5 + 312, *(_QWORD *)(v6 + 8));
  if ( v36 > 0x40 )
    sub_C43690((__int64)&v35, 0, 0);
  else
    v35 = 0;
  v7 = sub_BD45C0((unsigned __int8 *)v6, v5 + 312, (__int64)&v35, 1, 0, 0, 0, 0);
  v3 = 0;
  v8 = (__int64)v7;
  if ( *v7 == 61 )
  {
    v27 = *((_QWORD *)v7 - 4);
    v9 = sub_AE43F0(v5 + 312, *(_QWORD *)(v27 + 8));
    v10 = (unsigned __int8 *)v27;
    v38 = v9;
    if ( v9 > 0x40 )
    {
      sub_C43690((__int64)&v37, 0, 0);
      v10 = (unsigned __int8 *)v27;
    }
    else
    {
      v37 = 0;
    }
    if ( *sub_BD45C0(v10, v5 + 312, (__int64)&v37, 1, 0, 0, 0, 0) != 60 )
      goto LABEL_13;
    v29 = v38;
    if ( v38 > 0x40 )
    {
      v24 = v37;
      v19 = sub_C444A0((__int64)&v37);
      v3 = 0;
      v11 = (unsigned __int64)v24;
      if ( v29 - v19 > 0x40 )
      {
LABEL_16:
        if ( !v11 )
          goto LABEL_6;
LABEL_17:
        v28 = v3;
        j_j___libc_free_0_0(v11);
        v3 = v28;
        goto LABEL_6;
      }
      if ( *v24 )
        goto LABEL_17;
    }
    else
    {
      v3 = 0;
      if ( v37 )
        goto LABEL_6;
    }
    v12 = *(_QWORD *)(v8 + 40);
    v40 = 0;
    v39 = v8 + 24;
    v13 = sub_D319E0(v8, v12, &v39, 0, 0, 0, 0);
    if ( !v13 || (v14 = *(_QWORD *)(v13 + 8), *(_BYTE *)(v14 + 8) != 14) )
    {
LABEL_13:
      v3 = 0;
LABEL_14:
      if ( v38 <= 0x40 )
        goto LABEL_6;
      v11 = (unsigned __int64)v37;
      goto LABEL_16;
    }
    v30 = (unsigned __int8 *)v13;
    v15 = sub_AE43F0(v5 + 312, v14);
    v16 = v30;
    v42 = v15;
    if ( v15 > 0x40 )
    {
      sub_C43690((__int64)&v41, 0, 0);
      v16 = v30;
    }
    else
    {
      v41 = 0;
    }
    v17 = sub_BD45C0(v16, v5 + 312, (__int64)&v41, 1, 0, 0, 0, 0);
    if ( *v17 != 3
      || (v17[80] & 1) == 0
      || sub_B2FC80((__int64)v17)
      || (unsigned __int8)sub_B2F6B0((__int64)v17)
      || (v17[80] & 2) != 0 )
    {
      v18 = 0;
LABEL_29:
      v31 = v18;
      sub_969240((__int64 *)&v41);
      v3 = v31;
      goto LABEL_14;
    }
    v46 = v42;
    if ( v42 > 0x40 )
      sub_C43780((__int64)&v45, (const void **)&v41);
    else
      v45 = v41;
    sub_C45EE0((__int64)&v45, (__int64 *)&v35);
    v20 = (unsigned __int64)v45;
    v44 = v46;
    v43 = v45;
    if ( v46 > 0x40 )
    {
      v32 = v46;
      v25 = v45;
      if ( v32 - (unsigned int)sub_C444A0((__int64)&v43) > 0x40 )
        goto LABEL_45;
      v20 = *v25;
    }
    v21 = sub_E02A50((__int64)v17, v20, v5);
    v22 = v21;
    if ( v21 )
    {
      v33 = sub_29A3A40(a1, (__int64)v21, 0);
      if ( v33 )
      {
        sub_29A3E20(a1, v22, 0);
        v23 = v33;
LABEL_44:
        v34 = v23;
        sub_969240((__int64 *)&v43);
        v18 = v34;
        goto LABEL_29;
      }
    }
LABEL_45:
    v23 = 0;
    goto LABEL_44;
  }
LABEL_6:
  if ( v36 > 0x40 && v35 )
  {
    v26 = v3;
    j_j___libc_free_0_0(v35);
    return v26;
  }
  return v3;
}

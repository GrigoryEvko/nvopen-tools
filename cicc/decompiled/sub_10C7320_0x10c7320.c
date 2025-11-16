// Function: sub_10C7320
// Address: 0x10c7320
//
__int64 __fastcall sub_10C7320(__int64 *a1, __int64 a2)
{
  __int64 *v4; // rbx
  __int64 v7; // rdi
  bool v8; // zf
  unsigned __int8 v9; // al
  _BYTE *v10; // rbx
  char v11; // al
  __int64 v12; // rsi
  __int64 *v13; // rax
  __int64 v14; // r14
  unsigned int **v15; // r14
  const char *v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // rax
  unsigned __int8 v19; // r10
  __int64 v20; // r14
  char v21; // al
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  const char *v25; // rax
  _BYTE *v26; // rbx
  __int64 *v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rax
  _BYTE *v32; // rdi
  bool v33; // al
  __int64 v34; // rax
  unsigned __int8 v35; // [rsp+8h] [rbp-A8h]
  _BYTE *v36; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v37; // [rsp+17h] [rbp-99h]
  int v38; // [rsp+18h] [rbp-98h]
  unsigned __int8 v39; // [rsp+18h] [rbp-98h]
  _BYTE *v40; // [rsp+18h] [rbp-98h]
  _BYTE *v41; // [rsp+28h] [rbp-88h] BYREF
  _BYTE *v42; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v43; // [rsp+38h] [rbp-78h] BYREF
  _BYTE *v44; // [rsp+40h] [rbp-70h] BYREF
  int v45; // [rsp+48h] [rbp-68h]
  const char *v46; // [rsp+50h] [rbp-60h] BYREF
  __int64 *v47; // [rsp+58h] [rbp-58h]
  const char *v48; // [rsp+60h] [rbp-50h]
  _BYTE **v49; // [rsp+68h] [rbp-48h]
  __int16 v50; // [rsp+70h] [rbp-40h]

  v4 = (__int64 *)&v42;
  v46 = (const char *)&v41;
  v47 = (__int64 *)&v42;
  v48 = (const char *)&v41;
  v49 = &v42;
  if ( !sub_10C4D50((_QWORD **)&v46, (unsigned __int8 *)a2) )
    return 0;
  v7 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v7 + 16);
  v8 = !sub_BCAC40(v7, 1);
  v9 = *(_BYTE *)a2;
  v37 = *(_BYTE *)a2;
  if ( v8 )
  {
    v38 = 28;
  }
  else
  {
    v38 = 29;
    if ( v9 != 57 )
    {
      v38 = 28;
      if ( v9 == 86 && *(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL) == *(_QWORD *)(a2 + 8) )
      {
        v32 = *(_BYTE **)(a2 - 32);
        if ( *v32 <= 0x15u )
        {
          v33 = sub_AC30F0((__int64)v32);
          v37 = *(_BYTE *)a2;
          if ( v33 )
            v38 = 29;
        }
      }
    }
  }
  v43 = 0;
  v44 = 0;
  v46 = 0;
  v47 = (__int64 *)&v43;
  if ( *v41 == 59 )
  {
    v36 = v41;
    v21 = sub_995B10((_QWORD **)&v46, *((_QWORD *)v41 - 8));
    v22 = *((_QWORD *)v36 - 4);
    if ( v21 && v22 )
    {
      *v47 = v22;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10((_QWORD **)&v46, v22) )
        goto LABEL_11;
      v23 = *((_QWORD *)v36 - 8);
      if ( !v23 )
        goto LABEL_11;
      *v47 = v23;
    }
    if ( (unsigned __int8)sub_10C24F0(a1, v42, (unsigned __int8 *)a2) )
    {
      v41 = v43;
      goto LABEL_17;
    }
  }
LABEL_11:
  v10 = v42;
  v46 = 0;
  v47 = (__int64 *)&v44;
  if ( *v42 != 59 )
    return 0;
  v11 = sub_995B10((_QWORD **)&v46, *((_QWORD *)v42 - 8));
  v12 = *((_QWORD *)v10 - 4);
  if ( v11 && v12 )
  {
    *v47 = v12;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10((_QWORD **)&v46, v12) )
      return 0;
    v24 = *((_QWORD *)v10 - 8);
    if ( !v24 )
      return 0;
    *v47 = v24;
  }
  if ( !(unsigned __int8)sub_10C24F0(a1, v41, (unsigned __int8 *)a2) )
    return 0;
  v4 = (__int64 *)&v41;
  v42 = v44;
LABEL_17:
  v35 = sub_10C2350(a2, 0);
  if ( !v35 )
    return 0;
  v13 = sub_10BFAB0((__int64)a1, *v4, (unsigned __int8 *)a2);
  v14 = a1[4];
  *v4 = (__int64)v13;
  sub_B445D0((__int64)&v46, (char *)a2);
  sub_10BF960(v14, (__int64)v46, (__int16)v47);
  v15 = (unsigned int **)a1[4];
  if ( (unsigned int)v37 - 42 > 0x11 )
  {
    v25 = sub_BD5D20(a2);
    v26 = v42;
    v46 = v25;
    v50 = 773;
    v47 = v27;
    v48 = ".not";
    if ( v38 == 29 )
    {
      v40 = v41;
      v34 = sub_AD62B0(*((_QWORD *)v42 + 1));
      v30 = (__int64)v26;
      v29 = v34;
    }
    else
    {
      v40 = v41;
      v28 = sub_AD6530(*((_QWORD *)v42 + 1), (__int64)v41);
      v29 = (__int64)v26;
      v30 = v28;
    }
    v31 = sub_B36550(v15, (__int64)v40, v29, v30, (__int64)&v46, 0);
    v19 = v35;
    v20 = v31;
  }
  else
  {
    v16 = sub_BD5D20(a2);
    v50 = 773;
    v47 = v17;
    v46 = v16;
    v48 = ".not";
    v18 = sub_10BBE20((__int64 *)v15, v38, (__int64)v41, (__int64)v42, v45, 0, (__int64)&v46, 0);
    v19 = v35;
    v20 = v18;
  }
  v39 = v19;
  sub_F162A0((__int64)a1, a2, v20);
  sub_F16650((__int64)a1, v20, 0);
  return v39;
}

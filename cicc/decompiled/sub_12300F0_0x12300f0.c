// Function: sub_12300F0
// Address: 0x12300f0
//
__int64 __fastcall sub_12300F0(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned __int64 v5; // r13
  unsigned int v6; // r14d
  __int64 v8; // rsi
  bool v9; // zf
  char *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  char *v14; // rax
  const char *v15; // rax
  char v16; // dl
  _BYTE *v17; // r9
  __int64 v18; // rax
  __int64 v19; // r8
  unsigned __int64 v20; // rdx
  char *v21; // rax
  int v22; // eax
  __int64 v23; // rdx
  int v24; // ebx
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 *v30; // rax
  __int64 v31; // [rsp+0h] [rbp-3E0h]
  unsigned __int64 v32; // [rsp+10h] [rbp-3D0h]
  _BYTE *v33; // [rsp+10h] [rbp-3D0h]
  __int64 v34; // [rsp+18h] [rbp-3C8h]
  unsigned __int64 v35; // [rsp+28h] [rbp-3B8h] BYREF
  __int64 v36; // [rsp+30h] [rbp-3B0h] BYREF
  __int64 v37; // [rsp+38h] [rbp-3A8h] BYREF
  _BYTE *v38; // [rsp+40h] [rbp-3A0h] BYREF
  __int64 v39; // [rsp+48h] [rbp-398h] BYREF
  unsigned __int64 v40[4]; // [rsp+50h] [rbp-390h] BYREF
  char v41; // [rsp+70h] [rbp-370h]
  char v42; // [rsp+71h] [rbp-36Fh]
  __int64 v43; // [rsp+80h] [rbp-360h] BYREF
  char *v44; // [rsp+88h] [rbp-358h]
  __int64 v45; // [rsp+90h] [rbp-350h]
  int v46; // [rsp+98h] [rbp-348h]
  char v47; // [rsp+9Ch] [rbp-344h]
  char v48; // [rsp+A0h] [rbp-340h] BYREF
  const char *v49; // [rsp+1A0h] [rbp-240h] BYREF
  __int64 v50; // [rsp+1A8h] [rbp-238h]
  _BYTE v51[560]; // [rsp+1B0h] [rbp-230h] BYREF

  v5 = *(_QWORD *)(a1 + 232);
  v35 = 0;
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v36, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after switch condition") )
    return 1;
  if ( (unsigned __int8)sub_122FEA0(a1, &v37, &v35, a3) )
    return 1;
  v8 = 6;
  v6 = sub_120AFE0(a1, 6, "expected '[' with switch table");
  if ( (_BYTE)v6 )
    return 1;
  if ( *(_BYTE *)(*(_QWORD *)(v36 + 8) + 8LL) != 12 )
  {
    v51[17] = 1;
    v49 = "switch condition must have integer type";
    v51[16] = 3;
    sub_11FD800(a1 + 176, v5, (__int64)&v49, 1);
    return 1;
  }
  v9 = *(_DWORD *)(a1 + 240) == 7;
  v43 = 0;
  v44 = &v48;
  v49 = v51;
  v45 = 32;
  v46 = 0;
  v47 = 1;
  v50 = 0x2000000000LL;
  if ( v9 )
  {
LABEL_30:
    v22 = sub_1205200(a1 + 176);
    v23 = v37;
    v24 = v50;
    *(_DWORD *)(a1 + 240) = v22;
    v25 = v36;
    v34 = v23;
    v26 = sub_BD2DA0(80);
    v27 = v26;
    if ( v26 )
    {
      v8 = v25;
      sub_B53A60(v26, v25, v34, v24, 0, 0);
    }
    v28 = 0;
    v29 = 16LL * (unsigned int)v50;
    if ( (_DWORD)v50 )
    {
      do
      {
        v30 = (__int64 *)&v49[v28];
        v28 += 16;
        v8 = *v30;
        sub_B53E30(v27, *v30, v30[1]);
      }
      while ( v28 != v29 );
    }
    *a2 = v27;
  }
  else
  {
    while ( 1 )
    {
      v8 = (__int64)&v38;
      v32 = *(_QWORD *)(a1 + 232);
      if ( (unsigned __int8)sub_122FE20((__int64 **)a1, (__int64 *)&v38, a3) )
        break;
      v8 = 4;
      if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after case value") )
        break;
      v8 = (__int64)&v39;
      v40[0] = 0;
      if ( (unsigned __int8)sub_122FEA0(a1, &v39, v40, a3) )
        break;
      v8 = (__int64)v38;
      if ( !v47 )
        goto LABEL_25;
      v14 = v44;
      v11 = HIDWORD(v45);
      v10 = &v44[8 * HIDWORD(v45)];
      if ( v44 != v10 )
      {
        while ( v38 != *(_BYTE **)v14 )
        {
          v14 += 8;
          if ( v10 == v14 )
            goto LABEL_35;
        }
LABEL_17:
        v42 = 1;
        v15 = "duplicate case value in switch";
LABEL_18:
        v8 = v32;
        v40[0] = (unsigned __int64)v15;
        v41 = 3;
        sub_11FD800(a1 + 176, v32, (__int64)v40, 1);
        break;
      }
LABEL_35:
      if ( HIDWORD(v45) < (unsigned int)v45 )
      {
        ++HIDWORD(v45);
        *(_QWORD *)v10 = v38;
        ++v43;
      }
      else
      {
LABEL_25:
        sub_C8CC70((__int64)&v43, (__int64)v38, (__int64)v10, v11, v12, v13);
        if ( !v16 )
          goto LABEL_17;
      }
      v17 = v38;
      if ( *v38 != 17 )
      {
        v42 = 1;
        v15 = "case value is not a constant integer";
        goto LABEL_18;
      }
      v18 = (unsigned int)v50;
      v19 = v39;
      v20 = (unsigned int)v50 + 1LL;
      if ( v20 > HIDWORD(v50) )
      {
        v8 = (__int64)v51;
        v31 = v39;
        v33 = v38;
        sub_C8D5F0((__int64)&v49, v51, v20, 0x10u, v39, (__int64)v38);
        v18 = (unsigned int)v50;
        v19 = v31;
        v17 = v33;
      }
      v21 = (char *)&v49[16 * v18];
      *(_QWORD *)v21 = v17;
      *((_QWORD *)v21 + 1) = v19;
      LODWORD(v50) = v50 + 1;
      if ( *(_DWORD *)(a1 + 240) == 7 )
        goto LABEL_30;
    }
    v6 = 1;
  }
  if ( v49 != v51 )
    _libc_free(v49, v8);
  if ( !v47 )
    _libc_free(v44, v8);
  return v6;
}

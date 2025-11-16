// Function: sub_2415C20
// Address: 0x2415c20
//
void __fastcall sub_2415C20(__int64 *a1, unsigned int **a2, __int64 a3, _BYTE *a4)
{
  __int64 v4; // r15
  __int64 v7; // rax
  unsigned int v8; // eax
  __int64 *v9; // rax
  __int64 v10; // rax
  unsigned __int8 v11; // dl
  _BYTE **v12; // rax
  _BYTE *v13; // rax
  unsigned __int8 v14; // dl
  unsigned __int8 v15; // dl
  __int64 *v16; // rax
  char *v17; // rsi
  signed __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rax
  char *v22; // rsi
  signed __int64 v23; // rdx
  __int64 v24; // rax
  int v25; // ecx
  unsigned __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // r13
  __int64 *v29; // r14
  __int64 *v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rsi
  unsigned __int8 *v34; // rsi
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v38; // [rsp+8h] [rbp-98h]
  __int64 v39; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v40; // [rsp+18h] [rbp-88h]
  unsigned __int64 v41; // [rsp+20h] [rbp-80h] BYREF
  __int64 v42; // [rsp+28h] [rbp-78h]
  __int64 v43; // [rsp+30h] [rbp-70h]
  unsigned __int64 v44; // [rsp+40h] [rbp-60h] BYREF
  __int64 v45; // [rsp+48h] [rbp-58h]
  __int64 v46; // [rsp+50h] [rbp-50h]
  __int64 v47; // [rsp+58h] [rbp-48h]
  __int64 v48; // [rsp+60h] [rbp-40h] BYREF
  _BYTE v49[56]; // [rsp+68h] [rbp-38h] BYREF

  v4 = a3 + 48;
  v7 = sub_24159D0((__int64)a1, (__int64)a4);
  v40 = sub_2411210((__int64)a1, v7, (__int64)a2);
  if ( sub_B10CD0(v4) )
  {
    v8 = sub_B10CE0(v4);
    LODWORD(v45) = 32;
    v44 = v8;
    v9 = (__int64 *)sub_BD5C60(a3);
    v39 = sub_ACCFD0(v9, (__int64)&v44);
    if ( (unsigned int)v45 > 0x40 && v44 )
      j_j___libc_free_0_0(v44);
    LOWORD(v48) = 257;
    v10 = sub_B10CD0(v4);
    v11 = *(_BYTE *)(v10 - 16);
    if ( (v11 & 2) != 0 )
      v12 = *(_BYTE ***)(v10 - 32);
    else
      v12 = (_BYTE **)(v10 - 16 - 8LL * ((v11 >> 2) & 0xF));
    v13 = *v12;
    if ( *v13 == 16 )
      goto LABEL_10;
    v14 = *(v13 - 16);
    if ( (v14 & 2) != 0 )
    {
      v13 = (_BYTE *)**((_QWORD **)v13 - 4);
      if ( v13 )
      {
LABEL_10:
        v15 = *(v13 - 16);
        if ( (v15 & 2) != 0 )
          v16 = (__int64 *)*((_QWORD *)v13 - 4);
        else
          v16 = (__int64 *)&v13[-8 * ((v15 >> 2) & 0xF) - 16];
        v17 = (char *)*v16;
        if ( *v16 )
          v17 = (char *)sub_B91420(*v16);
        else
          v18 = 0;
        goto LABEL_14;
      }
    }
    else
    {
      v13 = *(_BYTE **)&v13[-8 * ((v14 >> 2) & 0xF) - 16];
      if ( v13 )
        goto LABEL_10;
    }
    v18 = 0;
    v17 = (char *)byte_3F871B3;
    goto LABEL_14;
  }
  LODWORD(v45) = 32;
  v44 = 0;
  v35 = (__int64 *)sub_BD5C60(a3);
  v39 = sub_ACCFD0(v35, (__int64)&v44);
  if ( (unsigned int)v45 > 0x40 && v44 )
    j_j___libc_free_0_0(v44);
  LOWORD(v48) = 257;
  v36 = *(_QWORD *)(sub_B43CB0(a3) + 40);
  v17 = *(char **)(v36 + 200);
  v18 = *(_QWORD *)(v36 + 208);
LABEL_14:
  v19 = sub_B33830((__int64)a2, v17, v18, (__int64)&v44, 0, 0, 1);
  LOWORD(v48) = 257;
  v20 = v19;
  v21 = sub_B43CB0(a3);
  v22 = (char *)sub_BD5D20(v21);
  v41 = 0;
  v38 = sub_B33830((__int64)a2, v22, v23, (__int64)&v44, 0, 0, 1);
  v42 = 0;
  v43 = 0;
  if ( (unsigned __int8)sub_240D530() )
  {
    v45 = sub_2414930(a1, a4);
    v46 = v20;
    v44 = (unsigned __int64)v40;
    v48 = v38;
    v47 = v39;
    sub_240DBB0((__int64)&v41, (char *)&v44, v49);
    v24 = *a1;
    v25 = v41;
    LOWORD(v48) = 257;
    v26 = *(_QWORD *)(v24 + 488);
    v27 = *(_QWORD *)(v24 + 496);
  }
  else
  {
    v45 = v20;
    v44 = (unsigned __int64)v40;
    v46 = v39;
    v47 = v38;
    sub_240DBB0((__int64)&v41, (char *)&v44, &v48);
    v32 = *a1;
    LOWORD(v48) = 257;
    v25 = v41;
    v26 = *(_QWORD *)(v32 + 472);
    v27 = *(_QWORD *)(v32 + 480);
  }
  v28 = sub_921880(a2, v26, v27, v25, (__int64)(v42 - v41) >> 3, (__int64)&v44, 0);
  v29 = (__int64 *)(v28 + 48);
  v30 = (__int64 *)sub_BD5C60(v28);
  *(_QWORD *)(v28 + 72) = sub_A7A090((__int64 *)(v28 + 72), v30, 1, 79);
  v31 = *(_QWORD *)(a3 + 48);
  v44 = v31;
  if ( !v31 )
  {
    if ( v29 == (__int64 *)&v44 )
      goto LABEL_20;
    v33 = *(_QWORD *)(v28 + 48);
    if ( !v33 )
      goto LABEL_20;
LABEL_27:
    sub_B91220(v28 + 48, v33);
    goto LABEL_28;
  }
  sub_B96E90((__int64)&v44, v31, 1);
  if ( v29 == (__int64 *)&v44 )
  {
    if ( v44 )
      sub_B91220((__int64)&v44, v44);
    goto LABEL_20;
  }
  v33 = *(_QWORD *)(v28 + 48);
  if ( v33 )
    goto LABEL_27;
LABEL_28:
  v34 = (unsigned __int8 *)v44;
  *(_QWORD *)(v28 + 48) = v44;
  if ( v34 )
    sub_B976B0((__int64)&v44, v34, v28 + 48);
LABEL_20:
  if ( v41 )
    j_j___libc_free_0(v41);
}

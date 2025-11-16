// Function: sub_1060DF0
// Address: 0x1060df0
//
__int64 __fastcall sub_1060DF0(
        unsigned int ***a1,
        unsigned int a2,
        __int64 a3,
        const void *a4,
        unsigned __int64 a5,
        __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // r8
  char v12; // r14
  const void *v13; // r10
  unsigned __int64 v14; // r11
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // rdx
  __int64 v17; // r9
  __int64 v18; // r8
  unsigned __int64 v19; // r8
  _QWORD *v20; // rax
  _QWORD *i; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int64 v24; // rsi
  __int64 v25; // r12
  _QWORD *v27; // rax
  _BYTE *v28; // rdx
  _QWORD *j; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r8
  __int64 v33; // rax
  __int64 v34; // rax
  _BYTE *v35; // rdi
  char v36; // [rsp+13h] [rbp-ADh]
  int v37; // [rsp+14h] [rbp-ACh]
  char nb; // [rsp+18h] [rbp-A8h]
  char na; // [rsp+18h] [rbp-A8h]
  char v42; // [rsp+20h] [rbp-A0h]
  int v43; // [rsp+20h] [rbp-A0h]
  char v44; // [rsp+20h] [rbp-A0h]
  const void *v45; // [rsp+20h] [rbp-A0h]
  int v48; // [rsp+38h] [rbp-88h]
  _QWORD *v49; // [rsp+38h] [rbp-88h]
  _BYTE *v50; // [rsp+50h] [rbp-70h] BYREF
  __int64 v51; // [rsp+58h] [rbp-68h]
  _BYTE v52[96]; // [rsp+60h] [rbp-60h] BYREF

  v7 = sub_B5A1E0(a2);
  v48 = v7;
  v8 = BYTE4(v7);
  v9 = sub_B5A3D0(a2);
  v11 = a5;
  v12 = v8;
  v13 = a4;
  v14 = HIDWORD(v9);
  v37 = v9;
  v50 = v52;
  v51 = 0x600000000LL;
  v15 = a5 + BYTE4(v9) + v8;
  if ( !BYTE4(v9) )
    LODWORD(v9) = a5;
  v16 = (unsigned int)v9;
  LODWORD(v9) = v48;
  if ( !v12 )
    LODWORD(v9) = a5;
  v9 = (unsigned int)v9;
  if ( (unsigned int)v9 > v16 )
    v9 = v16;
  if ( v9 >= a5 )
  {
    v17 = 8 * a5;
    v18 = (__int64)(8 * a5) >> 3;
    if ( 8 * a5 > 0x30 )
    {
      v36 = v14;
      sub_C8D5F0((__int64)&v50, v52, v18, 8u, v18, v17);
      v18 = (__int64)(8 * a5) >> 3;
      v17 = 8 * a5;
      v13 = a4;
      LOBYTE(v14) = v36;
      v35 = &v50[8 * (unsigned int)v51];
    }
    else
    {
      if ( !v17 )
        goto LABEL_10;
      v35 = v52;
    }
    nb = v14;
    v43 = v18;
    memcpy(v35, v13, v17);
    v17 = (unsigned int)v51;
    LOBYTE(v14) = nb;
    LODWORD(v18) = v43;
LABEL_10:
    LODWORD(v51) = v17 + v18;
    v19 = (unsigned int)(v17 + v18);
    if ( v19 != v15 )
    {
      if ( v19 <= v15 )
      {
        if ( v15 > HIDWORD(v51) )
        {
          v44 = v14;
          sub_C8D5F0((__int64)&v50, v52, v15, 8u, v19, v17);
          v19 = (unsigned int)v51;
          LOBYTE(v14) = v44;
        }
        v20 = &v50[8 * v19];
        for ( i = &v50[8 * v15]; i != v20; ++v20 )
        {
          if ( v20 )
            *v20 = 0;
        }
      }
      LODWORD(v51) = v15;
    }
LABEL_19:
    if ( !v12 )
      goto LABEL_20;
LABEL_41:
    v42 = v14;
    v33 = sub_1060D90(a1);
    *(_QWORD *)&v50[8 * v48] = v33;
    if ( !v42 )
      goto LABEL_21;
    goto LABEL_42;
  }
  if ( !v15 )
    goto LABEL_19;
  v27 = v52;
  v28 = v52;
  if ( v15 > 6 )
  {
    na = v14;
    v45 = v13;
    sub_C8D5F0((__int64)&v50, v52, v15, 8u, v11, v10);
    v28 = v50;
    LOBYTE(v14) = na;
    v13 = v45;
    v27 = &v50[8 * (unsigned int)v51];
  }
  for ( j = &v28[8 * v15]; j != v27; ++v27 )
  {
    if ( v27 )
      *v27 = 0;
  }
  LODWORD(v51) = v15;
  v30 = 0;
  v31 = 0;
  do
  {
    if ( (!v12 || v48 != v30) && (!(_BYTE)v14 || v37 != v30) )
    {
      v32 = *((_QWORD *)v13 + v31++);
      *(_QWORD *)&v50[8 * v30] = v32;
    }
    ++v30;
  }
  while ( v30 != v15 );
  if ( v12 )
    goto LABEL_41;
LABEL_20:
  if ( !(_BYTE)v14 )
    goto LABEL_21;
LABEL_42:
  v34 = sub_1060DB0((__int64)a1);
  *(_QWORD *)&v50[8 * v37] = v34;
LABEL_21:
  v49 = v50;
  v22 = sub_1060D40((__int64)a1);
  v23 = sub_B5B280(v22, a2, a3, v49);
  v24 = 0;
  if ( v23 )
    v24 = *(_QWORD *)(v23 + 24);
  v25 = sub_921880(*a1, v24, v23, (int)v50, v51, a6, 0);
  if ( v50 != v52 )
    _libc_free(v50, v24);
  return v25;
}

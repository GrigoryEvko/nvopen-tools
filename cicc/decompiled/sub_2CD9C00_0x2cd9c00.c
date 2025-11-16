// Function: sub_2CD9C00
// Address: 0x2cd9c00
//
__int64 __fastcall sub_2CD9C00(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 *v11; // r10
  __int64 *v12; // rbx
  __int64 *v14; // r14
  unsigned int v15; // r12d
  __int64 i; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 **v19; // rax
  __int64 **v20; // rdx
  bool v21; // cl
  __int64 v22; // rax
  __int64 *v23; // r11
  __int64 v24; // rsi
  _QWORD *v25; // r12
  _BYTE *v26; // rbx
  __int64 v27; // rax
  char *v28; // r13
  _QWORD *v29; // rbx
  __int64 v30; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned int v34; // r14d
  __int64 v35; // [rsp+0h] [rbp-4D0h]
  char v36; // [rsp+Bh] [rbp-4C5h]
  int v37; // [rsp+Ch] [rbp-4C4h]
  char v39; // [rsp+18h] [rbp-4B8h]
  __int64 v40; // [rsp+30h] [rbp-4A0h] BYREF
  __int64 v41; // [rsp+38h] [rbp-498h]
  __int64 v42; // [rsp+40h] [rbp-490h]
  unsigned int v43; // [rsp+48h] [rbp-488h]
  unsigned __int64 v44[2]; // [rsp+50h] [rbp-480h] BYREF
  char *v45; // [rsp+60h] [rbp-470h]
  __int64 v46; // [rsp+70h] [rbp-460h] BYREF
  char *v47; // [rsp+78h] [rbp-458h]
  __int64 v48; // [rsp+80h] [rbp-450h]
  int v49; // [rsp+88h] [rbp-448h]
  char v50; // [rsp+8Ch] [rbp-444h]
  char v51; // [rsp+90h] [rbp-440h] BYREF
  _BYTE *v52; // [rsp+190h] [rbp-340h] BYREF
  __int64 v53; // [rsp+198h] [rbp-338h]
  _BYTE v54[816]; // [rsp+1A0h] [rbp-330h] BYREF

  v47 = &v51;
  v52 = v54;
  v53 = 0x2000000000LL;
  v4 = *(_QWORD *)(*a1 + 80);
  v46 = 0;
  v48 = 32;
  if ( v4 )
    v4 -= 24;
  v49 = 0;
  v50 = 1;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v5 = sub_AA5190(v4);
  v8 = *a1;
  v9 = v5;
  v10 = v5 - 24;
  if ( v9 )
    v9 = v10;
  if ( (*(_BYTE *)(v8 + 2) & 1) != 0 )
  {
    sub_B2C6D0(*a1, a2, v6, v7);
    v11 = *(__int64 **)(v8 + 96);
    v12 = &v11[5 * *(_QWORD *)(v8 + 104)];
    if ( (*(_BYTE *)(v8 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v8, a2, v32, v33);
      v11 = *(__int64 **)(v8 + 96);
    }
  }
  else
  {
    v11 = *(__int64 **)(v8 + 96);
    v12 = &v11[5 * *(_QWORD *)(v8 + 104)];
  }
  if ( v11 != v12 )
  {
    v35 = v9;
    v14 = v11;
    while ( 1 )
    {
      v15 = sub_BD3610((__int64)v14, 0);
      if ( (_BYTE)v15 || *(_BYTE *)(v14[1] + 8) != 14 || !(unsigned __int8)sub_B2D680((__int64)v14) )
        goto LABEL_9;
      for ( i = v14[2]; i; i = *(_QWORD *)(i + 8) )
      {
        while ( 1 )
        {
          v17 = *(_QWORD *)(i + 24);
          if ( *(_BYTE *)v17 == 79 )
            break;
          i = *(_QWORD *)(i + 8);
          if ( !i )
            goto LABEL_21;
        }
        v18 = *(_QWORD *)(v17 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
          v18 = **(_QWORD **)(v18 + 16);
        if ( *(_DWORD *)(v18 + 8) >> 8 == 101 )
          goto LABEL_9;
      }
LABEL_21:
      if ( *(_BYTE *)(a2 + 28) )
      {
        v19 = *(__int64 ***)(a2 + 8);
        v20 = &v19[*(unsigned int *)(a2 + 20)];
        if ( v19 == v20 )
        {
LABEL_64:
          v21 = 0;
        }
        else
        {
          while ( v14 != *v19 )
          {
            if ( v20 == ++v19 )
              goto LABEL_64;
          }
          v21 = 1;
        }
      }
      else
      {
        v21 = sub_C8CA60(a2, (__int64)v14) != 0;
      }
      v39 = v21;
      v36 = sub_CE8660((__int64)v14);
      v37 = *(_DWORD *)(a1[2] + 8);
      v22 = sub_2CD67B0((__int64)a1, (unsigned __int64)v14, (__int64)&v46, v39);
      v23 = (__int64 *)v22;
      if ( !(_BYTE)qword_5013C48 && v37 > 69 && v36 )
      {
        v35 = sub_2CD6670(v22, v35);
        v23 = (__int64 *)v35;
      }
      if ( !v23 )
      {
        v34 = v15;
        goto LABEL_51;
      }
      if ( v14[1] == v23[1] )
      {
        v44[0] = (unsigned __int64)&v46;
        sub_BD79D0(v14, v23, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_2CD5F80, (__int64)v44);
LABEL_9:
        v14 += 5;
        if ( v12 == v14 )
          break;
      }
      else
      {
        v24 = (__int64)v14;
        v14 += 5;
        sub_2CD7710((__int64)a1, v24, (unsigned __int64)v23, (__int64)&v40, a3, (__int64)&v52);
        if ( v12 == v14 )
          break;
      }
    }
  }
  v25 = v52;
  v26 = &v52[24 * (unsigned int)v53];
  if ( v26 == v52 )
  {
    v34 = 1;
    goto LABEL_57;
  }
  do
  {
    while ( 1 )
    {
      v27 = v25[2];
      if ( v27 == 0 || v27 == -4096 || v27 == -8192 )
        goto LABEL_38;
      v44[0] = 6;
      v44[1] = 0;
      v45 = (char *)v25[2];
      v28 = v45;
      if ( v45 != 0 && v45 + 4096 != 0 && v45 != (char *)-8192LL )
      {
        sub_BD6050(v44, *v25 & 0xFFFFFFFFFFFFFFF8LL);
        v28 = v45;
      }
      if ( !v28 )
        goto LABEL_38;
      if ( v28 != (char *)-8192LL && v28 != (char *)-4096LL )
        sub_BD60C0(v44);
      if ( *v28 == 84 )
        break;
      v45 = 0;
      sub_F5CAB0(v28, 0, 0, (__int64)v44);
      if ( v45 )
        ((void (__fastcall *)(unsigned __int64 *, unsigned __int64 *, __int64))v45)(v44, v44, 3);
LABEL_38:
      v25 += 3;
      if ( v26 == (_BYTE *)v25 )
        goto LABEL_50;
    }
    v25 += 3;
    sub_F5CB10((__int64)v28, 0, 0);
  }
  while ( v26 != (_BYTE *)v25 );
LABEL_50:
  v34 = 1;
LABEL_51:
  v29 = v52;
  v25 = &v52[24 * (unsigned int)v53];
  if ( v52 != (_BYTE *)v25 )
  {
    do
    {
      v30 = *(v25 - 1);
      v25 -= 3;
      if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
        sub_BD60C0(v25);
    }
    while ( v29 != v25 );
    v25 = v52;
  }
LABEL_57:
  if ( v25 != (_QWORD *)v54 )
    _libc_free((unsigned __int64)v25);
  sub_C7D6A0(v41, 16LL * v43, 8);
  if ( !v50 )
    _libc_free((unsigned __int64)v47);
  return v34;
}

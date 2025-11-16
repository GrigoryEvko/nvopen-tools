// Function: sub_F55BE0
// Address: 0xf55be0
//
__int64 __fastcall sub_F55BE0(__int64 a1, unsigned __int8 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // r15
  unsigned int v10; // ebx
  __int64 v11; // r13
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // rax
  __int64 v17; // r15
  __int64 v18; // r13
  _QWORD *v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // r14
  unsigned __int8 *v23; // rsi
  __int64 v24; // rdx
  unsigned int v25; // r14d
  _QWORD *v26; // rdi
  __int64 v27; // r11
  __int64 v28; // rcx
  unsigned __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // r8
  __int64 v32; // rdx
  __int64 *v33; // r10
  __int64 v35; // rsi
  __int64 v36; // rbx
  __int64 *v37; // rcx
  __int64 *v38; // rax
  int v39; // [rsp+18h] [rbp-138h]
  __int64 v40; // [rsp+18h] [rbp-138h]
  __int64 *v41; // [rsp+18h] [rbp-138h]
  __int64 v42; // [rsp+20h] [rbp-130h]
  __int64 v43; // [rsp+20h] [rbp-130h]
  __int64 v45; // [rsp+30h] [rbp-120h] BYREF
  __int64 *v46; // [rsp+38h] [rbp-118h]
  __int64 v47; // [rsp+40h] [rbp-110h]
  int v48; // [rsp+48h] [rbp-108h]
  char v49; // [rsp+4Ch] [rbp-104h]
  char v50; // [rsp+50h] [rbp-100h] BYREF
  unsigned __int8 *v51; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v52; // [rsp+98h] [rbp-B8h]
  _BYTE v53[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a1 + 40);
  if ( a4 )
    sub_D72860(a4, a1, a3, (__int64)a4, a5, a6);
  v45 = 0;
  v46 = (__int64 *)&v50;
  v7 = *(_QWORD *)(v6 + 48);
  v47 = 8;
  v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
  v49 = 1;
  v48 = 0;
  v42 = v6 + 48;
  if ( v6 + 48 != v8 )
  {
    if ( !v8 )
      BUG();
    v9 = v8 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 <= 0xA )
    {
      v39 = sub_B46E30(v9);
      if ( v39 )
      {
        v10 = 0;
        while ( 1 )
        {
          while ( 1 )
          {
            v11 = sub_B46EC0(v9, v10);
            sub_AA5980(v11, v6, a2);
            if ( a3 )
              break;
LABEL_14:
            if ( v39 == ++v10 )
              goto LABEL_15;
          }
          if ( v49 )
          {
            v16 = v46;
            v12 = &v46[HIDWORD(v47)];
            if ( v46 != v12 )
            {
              while ( v11 != *v16 )
              {
                if ( v12 == ++v16 )
                  goto LABEL_46;
              }
              goto LABEL_14;
            }
LABEL_46:
            if ( HIDWORD(v47) >= (unsigned int)v47 )
              goto LABEL_44;
            ++v10;
            ++HIDWORD(v47);
            *v12 = v11;
            ++v45;
            if ( v39 == v10 )
              break;
          }
          else
          {
LABEL_44:
            ++v10;
            sub_C8CC70((__int64)&v45, v11, (__int64)v12, v13, v14, v15);
            if ( v39 == v10 )
              break;
          }
        }
      }
    }
  }
LABEL_15:
  v17 = a1 + 24;
  v18 = sub_BD5C60(a1);
  v19 = sub_BD2C40(72, unk_3F148B8);
  v22 = v19;
  if ( v19 )
    sub_B4C8A0((__int64)v19, v18, v17, 0);
  v23 = *(unsigned __int8 **)(a1 + 48);
  v51 = v23;
  if ( v23 )
  {
    sub_B96E90((__int64)&v51, (__int64)v23, 1);
    v24 = (__int64)(v22 + 6);
    if ( v22 + 6 == &v51 )
    {
      v23 = v51;
      if ( v51 )
        sub_B91220((__int64)&v51, (__int64)v51);
      goto LABEL_21;
    }
  }
  else
  {
    v24 = (__int64)(v22 + 6);
    if ( v22 + 6 == &v51 )
      goto LABEL_21;
  }
  v35 = v22[6];
  if ( v35 )
  {
    v40 = v24;
    sub_B91220(v24, v35);
    v24 = v40;
  }
  v23 = v51;
  v22[6] = v51;
  if ( v23 )
    sub_B976B0((__int64)&v51, v23, v24);
LABEL_21:
  v25 = 0;
  if ( v17 != v42 )
  {
    while ( 1 )
    {
      if ( *(_QWORD *)(v17 - 8) )
      {
        v23 = (unsigned __int8 *)sub_ACADE0(*(__int64 ***)(v17 - 16));
        sub_BD84D0(v17 - 24, (__int64)v23);
      }
      v26 = (_QWORD *)(v17 - 24);
      v17 = *(_QWORD *)(v17 + 8);
      ++v25;
      sub_B43D60(v26);
      if ( v17 == v42 )
        break;
      if ( !v17 )
        BUG();
    }
  }
  if ( !a3 )
    goto LABEL_36;
  v27 = 0;
  v28 = 0;
  v29 = (unsigned int)(HIDWORD(v47) - v48);
  v51 = v53;
  v52 = 0x800000000LL;
  if ( v29 > 8 )
  {
    sub_C8D5F0((__int64)&v51, v53, v29, 0x10u, v20, v21);
    v27 = (unsigned int)v52;
    v30 = v46;
    v28 = (unsigned int)v52;
    if ( !v49 )
      goto LABEL_30;
LABEL_49:
    v31 = (__int64)&v30[HIDWORD(v47)];
    goto LABEL_31;
  }
  v30 = v46;
  if ( v49 )
    goto LABEL_49;
LABEL_30:
  v31 = (__int64)&v30[(unsigned int)v47];
LABEL_31:
  if ( v30 != (__int64 *)v31 )
  {
    while ( 1 )
    {
      v32 = *v30;
      v33 = v30;
      if ( (unsigned __int64)*v30 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( (__int64 *)v31 == ++v30 )
        goto LABEL_34;
    }
    if ( (__int64 *)v31 != v30 )
    {
      do
      {
        v36 = v32 | 4;
        if ( v28 + 1 > (unsigned __int64)HIDWORD(v52) )
        {
          v41 = v33;
          v43 = v31;
          sub_C8D5F0((__int64)&v51, v53, v28 + 1, 0x10u, v31, v21);
          v28 = (unsigned int)v52;
          v33 = v41;
          v31 = v43;
        }
        v37 = (__int64 *)&v51[16 * v28];
        *v37 = v6;
        v37[1] = v36;
        v28 = (unsigned int)(v52 + 1);
        v38 = v33 + 1;
        LODWORD(v52) = v52 + 1;
        if ( v33 + 1 == (__int64 *)v31 )
          break;
        while ( 1 )
        {
          v32 = *v38;
          v33 = v38;
          if ( (unsigned __int64)*v38 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( (__int64 *)v31 == ++v38 )
            goto LABEL_58;
        }
      }
      while ( (__int64 *)v31 != v38 );
LABEL_58:
      v27 = (unsigned int)v28;
    }
  }
LABEL_34:
  v23 = v51;
  sub_FFB3D0(a3, v51, v27);
  if ( v51 != v53 )
    _libc_free(v51, v23);
LABEL_36:
  sub_AA6320(v6);
  if ( !v49 )
    _libc_free(v46, v23);
  return v25;
}

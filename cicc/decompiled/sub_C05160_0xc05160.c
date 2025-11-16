// Function: sub_C05160
// Address: 0xc05160
//
void __fastcall sub_C05160(__int64 a1, _BYTE *a2)
{
  char v2; // cl
  _QWORD *v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 *v9; // rax
  __int64 *v10; // rdx
  char v11; // dl
  __int64 v12; // r8
  _BYTE *v13; // r11
  __int64 v14; // r15
  char v15; // al
  __int64 v16; // r12
  _BYTE *v17; // rax
  __int64 v18; // rax
  int v19; // ecx
  _QWORD *v20; // rsi
  _BYTE *v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 *v26; // rax
  __int64 v27; // rax
  _BYTE *v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rcx
  int v31; // edx
  unsigned __int64 v32; // rax
  _BYTE *v33; // rax
  bool v34; // zf
  __int64 v35; // rax
  _QWORD *v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // [rsp+0h] [rbp-130h]
  __int64 v41; // [rsp+0h] [rbp-130h]
  __int64 v42; // [rsp+0h] [rbp-130h]
  __int64 v43; // [rsp+0h] [rbp-130h]
  _BYTE *v44; // [rsp+8h] [rbp-128h]
  _BYTE *v45; // [rsp+8h] [rbp-128h]
  _BYTE *v46; // [rsp+8h] [rbp-128h]
  _BYTE *v47; // [rsp+10h] [rbp-120h]
  _BYTE *v48; // [rsp+18h] [rbp-118h]
  __int64 v49[4]; // [rsp+20h] [rbp-110h] BYREF
  char v50; // [rsp+40h] [rbp-F0h]
  char v51; // [rsp+41h] [rbp-EFh]
  _QWORD *v52; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v53; // [rsp+58h] [rbp-D8h]
  _QWORD v54[8]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+A0h] [rbp-90h] BYREF
  __int64 *v56; // [rsp+A8h] [rbp-88h]
  __int64 v57; // [rsp+B0h] [rbp-80h]
  int v58; // [rsp+B8h] [rbp-78h]
  char v59; // [rsp+BCh] [rbp-74h]
  char v60; // [rsp+C0h] [rbp-70h] BYREF

  v2 = 1;
  v5 = v54;
  v52 = v54;
  v54[0] = a2;
  v55 = 0;
  v57 = 8;
  v58 = 0;
  v59 = 1;
  v48 = 0;
  v47 = 0;
  v53 = 0x800000001LL;
  v56 = (__int64 *)&v60;
  v6 = 1;
  while ( 1 )
  {
    v7 = v6;
    v8 = v5[v6 - 1];
    LODWORD(v53) = v6 - 1;
    if ( !v2 )
      goto LABEL_11;
    v9 = v56;
    v10 = &v56[HIDWORD(v57)];
    if ( v56 != v10 )
    {
      while ( v8 != *v9 )
      {
        if ( v10 == ++v9 )
          goto LABEL_10;
      }
LABEL_7:
      v7 = (__int64)v49;
      v51 = 1;
      v49[0] = (__int64)"FuncletPadInst must not be nested within itself";
      v50 = 3;
      sub_BDBF70((__int64 *)a1, (__int64)v49);
      if ( *(_QWORD *)a1 && v8 )
      {
LABEL_9:
        v7 = v8;
        sub_BDBD80(a1, (_BYTE *)v8);
      }
LABEL_20:
      if ( !v59 )
        goto LABEL_71;
      goto LABEL_21;
    }
LABEL_10:
    if ( HIDWORD(v57) < (unsigned int)v57 )
    {
      ++HIDWORD(v57);
      *v10 = v8;
      ++v55;
    }
    else
    {
LABEL_11:
      v7 = v8;
      sub_C8CC70(&v55, v8);
      if ( !v11 )
        goto LABEL_7;
    }
    v12 = *(_QWORD *)(v8 + 16);
    if ( v12 )
      break;
LABEL_40:
    v6 = v53;
    if ( !(_DWORD)v53 )
    {
LABEL_93:
      if ( !v47 )
        goto LABEL_95;
      v8 = *((_QWORD *)a2 - 4);
      if ( *(_BYTE *)v8 != 39 )
        goto LABEL_95;
      if ( (*(_BYTE *)(v8 + 2) & 1) != 0 && (v37 = *(_QWORD *)(*(_QWORD *)(v8 - 8) + 32LL)) != 0 )
      {
        v38 = sub_BDB910(v37);
        if ( v38 )
          v38 -= 24;
      }
      else
      {
        v39 = (__int64 *)sub_BD5C60((__int64)a2);
        v38 = sub_AC3540(v39);
      }
      if ( (_BYTE *)v38 == v47 )
      {
LABEL_95:
        v7 = (__int64)a2;
        sub_BF6FE0(a1, (__int64)a2);
        if ( !v59 )
          goto LABEL_71;
        goto LABEL_21;
      }
      v7 = (__int64)v49;
      v51 = 1;
      v49[0] = (__int64)"Unwind edges out of a catch must have the same unwind dest as the parent catchswitch";
      v50 = 3;
      sub_BDBF70((__int64 *)a1, (__int64)v49);
      if ( *(_QWORD *)a1 )
      {
        sub_BDBD80(a1, a2);
        if ( v48 )
          sub_BDBD80(a1, v48);
        goto LABEL_9;
      }
      goto LABEL_20;
    }
    v5 = v52;
    v2 = v59;
  }
  v13 = 0;
  while ( 1 )
  {
    v14 = *(_QWORD *)(v12 + 24);
    v15 = *(_BYTE *)v14;
    if ( *(_BYTE *)v14 <= 0x1Cu )
      break;
    switch ( v15 )
    {
      case '%':
        if ( (*(_BYTE *)(v14 + 2) & 1) == 0 )
          goto LABEL_49;
        v25 = *(_QWORD *)(v14 + 32 * (1LL - (*(_DWORD *)(v14 + 4) & 0x7FFFFFF)));
LABEL_48:
        if ( !v25 )
          goto LABEL_49;
        goto LABEL_57;
      case '\'':
        if ( (*(_BYTE *)(v14 + 2) & 1) == 0 )
          goto LABEL_30;
        v25 = *(_QWORD *)(*(_QWORD *)(v14 - 8) + 32LL);
        if ( !v25 )
        {
LABEL_49:
          v40 = v12;
          v26 = (__int64 *)sub_BD5C60((__int64)a2);
          v27 = sub_AC3540(v26);
          v12 = v40;
          v13 = a2;
          v28 = (_BYTE *)v27;
LABEL_50:
          if ( v48 )
          {
            if ( v47 != v28 )
            {
              v7 = (__int64)v49;
              v51 = 1;
              v49[0] = (__int64)"Unwind edges out of a funclet pad must have the same unwind dest";
              v50 = 3;
              sub_BDBF70((__int64 *)a1, (__int64)v49);
              if ( *(_QWORD *)a1 )
              {
                sub_BDBD80(a1, a2);
                sub_BDBD80(a1, (_BYTE *)v14);
                v7 = (__int64)v48;
                sub_BDBD80(a1, v48);
              }
              goto LABEL_20;
            }
          }
          else
          {
            v48 = (_BYTE *)v14;
            v47 = v28;
            if ( *a2 == 80 && *v28 != 21 )
            {
              if ( (unsigned __int8)(*v28 - 80) > 1u )
                v35 = **((_QWORD **)v28 - 1);
              else
                v35 = *((_QWORD *)v28 - 4);
              if ( v35 == *((_QWORD *)a2 - 4) )
              {
                v7 = (__int64)v49;
                v43 = v12;
                v46 = v13;
                v47 = v28;
                v49[0] = (__int64)a2;
                v36 = (_QWORD *)sub_C04EB0(a1 + 864, v49);
                v48 = (_BYTE *)v14;
                v13 = v46;
                *v36 = v14;
                v12 = v43;
              }
              else
              {
                v48 = (_BYTE *)v14;
                v47 = v28;
              }
            }
          }
          goto LABEL_52;
        }
LABEL_57:
        v41 = v12;
        v44 = v13;
        v29 = sub_AA4FF0(v25);
        v30 = v29;
        if ( !v29 )
          BUG();
        v13 = v44;
        v12 = v41;
        v31 = *(unsigned __int8 *)(v29 - 24);
        v32 = (unsigned int)(v31 - 39);
        if ( (unsigned int)v32 > 0x38 )
          goto LABEL_30;
        v7 = 0x100060000000001LL;
        if ( !_bittest64(&v7, v32) )
          goto LABEL_30;
        if ( (unsigned __int8)(v31 - 80) <= 1u )
        {
          v7 = *(_QWORD *)(v30 - 56);
LABEL_62:
          if ( v8 == v7 )
            goto LABEL_30;
          goto LABEL_63;
        }
        v7 = **(_QWORD **)(v30 - 32);
        if ( v7 )
          goto LABEL_62;
LABEL_63:
        v33 = (_BYTE *)v8;
        while ( 1 )
        {
          if ( a2 == v33 )
          {
            v28 = (_BYTE *)(v30 - 24);
            v13 = a2;
            goto LABEL_50;
          }
          v33 = (_BYTE *)((unsigned __int8)(*v33 - 80) <= 1u ? *((_QWORD *)v33 - 4) : **((_QWORD **)v33 - 1));
          if ( v33 == (_BYTE *)v7 )
            break;
          if ( *v33 == 21 )
            goto LABEL_52;
        }
        v13 = v33;
LABEL_52:
        if ( a2 != (_BYTE *)v8 )
          goto LABEL_31;
        v12 = *(_QWORD *)(v12 + 8);
        if ( !v12 )
          goto LABEL_31;
        break;
      case '"':
        v25 = *(_QWORD *)(v14 - 64);
        goto LABEL_48;
      case 'U':
LABEL_30:
        v12 = *(_QWORD *)(v12 + 8);
        if ( !v12 )
          goto LABEL_31;
        break;
      case 'P':
        v23 = (unsigned int)v53;
        v24 = (unsigned int)v53 + 1LL;
        if ( v24 > HIDWORD(v53) )
        {
          v7 = (__int64)v54;
          v42 = v12;
          v45 = v13;
          sub_C8D5F0(&v52, v54, v24, 8);
          v23 = (unsigned int)v53;
          v12 = v42;
          v13 = v45;
        }
        v52[v23] = v14;
        LODWORD(v53) = v53 + 1;
        v12 = *(_QWORD *)(v12 + 8);
        if ( !v12 )
        {
LABEL_31:
          if ( !v13 || v13 == (_BYTE *)v8 )
            goto LABEL_40;
          v19 = v53;
          if ( (_DWORD)v53 )
          {
            v20 = &v52[(unsigned int)v53];
            v21 = (_BYTE *)*(v20 - 1);
            if ( (unsigned __int8)(*v21 - 80) <= 1u )
            {
LABEL_35:
              v22 = *((_QWORD *)v21 - 4);
              goto LABEL_36;
            }
            while ( 1 )
            {
              v22 = **((_QWORD **)v21 - 1);
LABEL_36:
              while ( v8 != v22 )
              {
                while ( (unsigned __int8)(*(_BYTE *)v8 - 80) > 1u )
                {
                  v8 = **(_QWORD **)(v8 - 8);
                  if ( v8 )
                    goto LABEL_39;
                  if ( !v22 )
                    goto LABEL_86;
                }
                v8 = *(_QWORD *)(v8 - 32);
LABEL_39:
                if ( v13 == (_BYTE *)v8 )
                  goto LABEL_40;
              }
LABEL_86:
              --v19;
              --v20;
              LODWORD(v53) = v19;
              if ( !v19 )
                break;
              v8 = v22;
              v21 = (_BYTE *)*(v20 - 1);
              if ( (unsigned __int8)(*v21 - 80) <= 1u )
                goto LABEL_35;
            }
          }
          goto LABEL_93;
        }
        break;
      case '&':
        goto LABEL_30;
      default:
        goto LABEL_15;
    }
  }
LABEL_15:
  v16 = *(_QWORD *)a1;
  v51 = 1;
  v49[0] = (__int64)"Bogus funclet pad use";
  v50 = 3;
  if ( v16 )
  {
    v7 = v16;
    sub_CA0E80(v49, v16);
    v17 = *(_BYTE **)(v16 + 32);
    if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 24) )
    {
      v7 = 10;
      sub_CB5D20(v16, 10);
    }
    else
    {
      *(_QWORD *)(v16 + 32) = v17 + 1;
      *v17 = 10;
    }
    v18 = *(_QWORD *)a1;
    *(_BYTE *)(a1 + 152) = 1;
    if ( v18 )
    {
      v7 = v14;
      sub_BDBD80(a1, (_BYTE *)v14);
    }
    goto LABEL_20;
  }
  v34 = v59 == 0;
  *(_BYTE *)(a1 + 152) = 1;
  if ( !v34 )
    goto LABEL_21;
LABEL_71:
  _libc_free(v56, v7);
LABEL_21:
  if ( v52 != v54 )
    _libc_free(v52, v7);
}

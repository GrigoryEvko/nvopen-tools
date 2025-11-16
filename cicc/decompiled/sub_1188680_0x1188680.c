// Function: sub_1188680
// Address: 0x1188680
//
unsigned __int8 *__fastcall sub_1188680(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  _BYTE *v3; // r14
  __int64 v4; // r13
  int v8; // eax
  _BYTE *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r14
  bool v14; // r15
  _BYTE **v15; // r10
  __int16 *v16; // rdx
  _QWORD *v17; // rax
  __int64 *v18; // r10
  _QWORD *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // rdx
  unsigned __int8 v26; // al
  __int64 v27; // rsi
  char v28; // al
  char v29; // r14
  char v30; // al
  _BYTE *v31; // rdi
  unsigned int v32; // r15d
  bool v33; // al
  __int64 v34; // rdx
  _BYTE *v35; // rax
  unsigned int v36; // r15d
  unsigned int v37; // ecx
  __int64 v38; // rax
  unsigned int v39; // ecx
  unsigned int v40; // r15d
  int v41; // eax
  int v42; // [rsp+4h] [rbp-10Ch]
  char v43; // [rsp+8h] [rbp-108h]
  __int64 v44; // [rsp+10h] [rbp-100h]
  __int64 *v45; // [rsp+10h] [rbp-100h]
  __int64 *v46; // [rsp+10h] [rbp-100h]
  _BYTE **v47; // [rsp+10h] [rbp-100h]
  _BYTE **v48; // [rsp+10h] [rbp-100h]
  __int64 v49; // [rsp+10h] [rbp-100h]
  unsigned int v50; // [rsp+10h] [rbp-100h]
  _BYTE **v51; // [rsp+10h] [rbp-100h]
  __int64 v52; // [rsp+18h] [rbp-F8h]
  _QWORD v53[2]; // [rsp+20h] [rbp-F0h] BYREF
  _BYTE *v54; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v55; // [rsp+38h] [rbp-D8h]
  _BYTE v56[64]; // [rsp+40h] [rbp-D0h] BYREF
  const char *v57; // [rsp+80h] [rbp-90h] BYREF
  __int16 *v58; // [rsp+88h] [rbp-88h]
  __int64 v59; // [rsp+90h] [rbp-80h]
  int v60; // [rsp+98h] [rbp-78h]
  char v61; // [rsp+9Ch] [rbp-74h]
  __int16 v62; // [rsp+A0h] [rbp-70h] BYREF

  v2 = *(_QWORD *)(a1 - 96);
  v3 = *(_BYTE **)(a1 - 64);
  v4 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v2 != 82 )
    return 0;
  v44 = *(_QWORD *)(v2 - 64);
  if ( !v44 )
    return 0;
  v43 = sub_1178DE0(*(_QWORD *)(v2 - 32));
  if ( !v43 )
    return 0;
  v8 = sub_B53900(v2);
  if ( (unsigned int)(v8 - 32) > 1 )
    return 0;
  if ( v8 == 33 )
  {
    v9 = v3;
    v3 = (_BYTE *)v4;
    v4 = (__int64)v9;
  }
  if ( *v3 > 0x15u || *(_BYTE *)v4 != 46 )
    return 0;
  v10 = *(_QWORD *)(v4 - 64);
  v52 = *(_QWORD *)(v4 - 32);
  if ( v44 == v10 )
  {
    if ( !*(_QWORD *)(v4 - 32) )
      return 0;
  }
  else
  {
    if ( v52 != v44 || !v10 )
      return 0;
    v52 = *(_QWORD *)(v4 - 64);
  }
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    v11 = *(_QWORD *)(v2 - 8);
  else
    v11 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  v12 = sub_AD7180(v3, *(unsigned __int8 **)(v11 + 32));
  v13 = v12;
  if ( !v12 )
  {
    v26 = MEMORY[0];
LABEL_28:
    if ( (unsigned __int8)(v26 - 12) <= 1u )
      goto LABEL_18;
    if ( (unsigned __int8)(v26 - 9) <= 2u )
    {
      v27 = v13;
      v57 = 0;
      v54 = v56;
      v53[1] = &v54;
      v58 = &v62;
      v59 = 8;
      v60 = 0;
      v61 = 1;
      v55 = 0x800000000LL;
      v53[0] = &v57;
      v28 = sub_AA8FD0(v53, v13);
      v15 = &v54;
      v29 = v28;
      if ( v28 )
      {
        while ( 1 )
        {
          v31 = v54;
          if ( !(_DWORD)v55 )
            break;
          v47 = v15;
          v27 = *(_QWORD *)&v54[8 * (unsigned int)v55 - 8];
          LODWORD(v55) = v55 - 1;
          v30 = sub_AA8FD0(v53, v27);
          v15 = v47;
          if ( !v30 )
            goto LABEL_51;
        }
      }
      else
      {
LABEL_51:
        v31 = v54;
        v29 = 0;
      }
      if ( v31 != v56 )
      {
        v48 = v15;
        _libc_free(v31, v27);
        v15 = v48;
      }
      if ( !v61 )
      {
        v51 = v15;
        _libc_free(v58, v27);
        v15 = v51;
      }
      if ( v29 )
        goto LABEL_19;
    }
    return 0;
  }
  v14 = sub_AC30F0(v12);
  if ( !v14 )
  {
    v26 = *(_BYTE *)v13;
    if ( *(_BYTE *)v13 == 17 )
    {
      v32 = *(_DWORD *)(v13 + 32);
      if ( v32 <= 0x40 )
        v33 = *(_QWORD *)(v13 + 24) == 0;
      else
        v33 = v32 == (unsigned int)sub_C444A0(v13 + 24);
LABEL_44:
      v15 = &v54;
      if ( v33 )
        goto LABEL_19;
      goto LABEL_45;
    }
    v34 = *(_QWORD *)(v13 + 8);
    v49 = v34;
    if ( (unsigned int)*(unsigned __int8 *)(v34 + 8) - 17 > 1 )
      goto LABEL_28;
    v35 = sub_AD7630(v13, 0, v34);
    if ( v35 && *v35 == 17 )
    {
      v36 = *((_DWORD *)v35 + 8);
      if ( v36 > 0x40 )
      {
        v33 = v36 == (unsigned int)sub_C444A0((__int64)(v35 + 24));
        goto LABEL_44;
      }
      v15 = &v54;
      if ( !*((_QWORD *)v35 + 3) )
        goto LABEL_19;
    }
    else if ( *(_BYTE *)(v49 + 8) == 17 )
    {
      v37 = 0;
      v42 = *(_DWORD *)(v49 + 32);
      if ( v42 )
      {
        while ( 1 )
        {
          v50 = v37;
          v38 = sub_AD69F0((unsigned __int8 *)v13, v37);
          v39 = v50;
          if ( !v38 )
            break;
          if ( *(_BYTE *)v38 != 13 )
          {
            if ( *(_BYTE *)v38 != 17 )
              break;
            v40 = *(_DWORD *)(v38 + 32);
            if ( v40 <= 0x40 )
            {
              if ( *(_QWORD *)(v38 + 24) )
                break;
            }
            else
            {
              v41 = sub_C444A0(v38 + 24);
              v39 = v50;
              if ( v40 != v41 )
                break;
            }
            v14 = v43;
          }
          v37 = v39 + 1;
          if ( v42 == v37 )
          {
            v15 = &v54;
            if ( v14 )
              goto LABEL_19;
            break;
          }
        }
      }
    }
LABEL_45:
    v26 = *(_BYTE *)v13;
    goto LABEL_28;
  }
LABEL_18:
  v15 = &v54;
LABEL_19:
  v45 = (__int64 *)v15;
  v57 = sub_BD5D20(v52);
  v59 = (__int64)".fr";
  v62 = 773;
  v58 = v16;
  v17 = sub_BD2C40(72, unk_3F10A14);
  v18 = v45;
  v19 = v17;
  if ( v17 )
  {
    sub_B549F0((__int64)v17, v52, (__int64)&v57, 0, 0);
    v18 = v45;
  }
  v46 = v18;
  sub_B44220(v19, v4 + 24, 0);
  v20 = *(_QWORD *)(a2 + 40);
  v54 = v19;
  sub_1187E30(v20 + 2096, v46, v21, v22, v23, v24);
  if ( (*(_BYTE *)(v4 + 7) & 0x40) != 0 )
    v25 = *(_QWORD **)(v4 - 8);
  else
    v25 = (_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
  sub_F20660(a2, v4, *v25 != v52, (__int64)v19);
  return sub_F162A0(a2, a1, v4);
}

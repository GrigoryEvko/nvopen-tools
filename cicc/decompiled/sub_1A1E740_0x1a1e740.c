// Function: sub_1A1E740
// Address: 0x1a1e740
//
__int64 __fastcall sub_1A1E740(__int64 *a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 v4; // r14
  _QWORD *v5; // r15
  _BYTE *v6; // rbx
  int v7; // r8d
  int v8; // r9d
  char v9; // dl
  _QWORD *v10; // r13
  _QWORD *v11; // rsi
  _QWORD *v12; // rcx
  __int64 v13; // rax
  _QWORD *v14; // rax
  int v15; // eax
  _BYTE *v16; // rdi
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  __int64 **v19; // rdx
  __int64 v20; // r14
  __int64 *v21; // rcx
  char v22; // al
  __int64 *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // rsi
  __int64 v27; // rax
  bool v28; // al
  __int64 v29; // rsi
  __int64 v30; // rdi
  unsigned int v31; // r8d
  _QWORD *v32; // rax
  __int64 v33; // rcx
  __int64 v35; // r15
  int v36; // r8d
  int v37; // r9d
  char v38; // dl
  _QWORD *v39; // rbx
  _QWORD *v40; // rax
  _QWORD *v41; // rsi
  _QWORD *v42; // rcx
  __int64 v43; // rax
  __int64 *v44; // rax
  int v45; // [rsp+0h] [rbp-F0h]
  __int64 v48; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE *v49; // [rsp+28h] [rbp-C8h]
  _BYTE *v50; // [rsp+30h] [rbp-C0h]
  __int64 v51; // [rsp+38h] [rbp-B8h]
  int v52; // [rsp+40h] [rbp-B0h]
  _BYTE v53[40]; // [rsp+48h] [rbp-A8h] BYREF
  _BYTE *v54; // [rsp+70h] [rbp-80h] BYREF
  __int64 v55; // [rsp+78h] [rbp-78h]
  _BYTE v56[112]; // [rsp+80h] [rbp-70h] BYREF

  v4 = *(_QWORD *)(a2 + 8);
  v54 = v56;
  v48 = 0;
  v49 = v53;
  v50 = v53;
  v51 = 4;
  v52 = 0;
  v55 = 0x400000000LL;
  if ( !v4 )
  {
    *a3 = 0;
    v20 = 0;
    goto LABEL_51;
  }
  v5 = v53;
  v6 = v53;
  while ( 1 )
  {
    v10 = sub_1648700(v4);
    if ( v5 == (_QWORD *)v6 )
    {
      v11 = &v5[HIDWORD(v51)];
      if ( v11 != v5 )
      {
        v12 = 0;
        while ( v10 != (_QWORD *)*v5 )
        {
          if ( *v5 == -2 )
            v12 = v5;
          if ( v11 == ++v5 )
          {
            if ( !v12 )
              goto LABEL_54;
            *v12 = v10;
            --v52;
            ++v48;
            goto LABEL_15;
          }
        }
        goto LABEL_4;
      }
LABEL_54:
      if ( HIDWORD(v51) < (unsigned int)v51 )
        break;
    }
    sub_16CCBA0((__int64)&v48, (__int64)v10);
    if ( v9 )
    {
LABEL_15:
      v13 = (unsigned int)v55;
      if ( (unsigned int)v55 < HIDWORD(v55) )
        goto LABEL_16;
      goto LABEL_56;
    }
LABEL_4:
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      goto LABEL_17;
LABEL_5:
    v6 = v50;
    v5 = v49;
  }
  ++HIDWORD(v51);
  *v11 = v10;
  v13 = (unsigned int)v55;
  ++v48;
  if ( (unsigned int)v55 < HIDWORD(v55) )
    goto LABEL_16;
LABEL_56:
  sub_16CD150((__int64)&v54, v56, 0, 16, v7, v8);
  v13 = (unsigned int)v55;
LABEL_16:
  v14 = &v54[16 * v13];
  v14[1] = v10;
  *v14 = a2;
  LODWORD(v55) = v55 + 1;
  v4 = *(_QWORD *)(v4 + 8);
  if ( v4 )
    goto LABEL_5;
LABEL_17:
  v15 = v55;
  v16 = v54;
  *a3 = 0;
  if ( !v15 )
  {
LABEL_76:
    v20 = 0;
    goto LABEL_49;
  }
  while ( 2 )
  {
    v19 = (__int64 **)&v16[16 * v15 - 16];
    v20 = (__int64)v19[1];
    v21 = *v19;
    LODWORD(v55) = v15 - 1;
    v22 = *(_BYTE *)(v20 + 16);
    if ( v22 == 54 )
    {
      v17 = *(_QWORD *)v20;
      goto LABEL_20;
    }
    if ( v22 == 55 )
    {
      v23 = *(__int64 **)(v20 - 48);
      if ( v23 == v21 )
        goto LABEL_49;
      v17 = *v23;
LABEL_20:
      v18 = (unsigned __int64)(sub_127FA20(*a1, v17) + 7) >> 3;
      if ( v18 < *a3 )
        v18 = *a3;
      *a3 = v18;
LABEL_23:
      v15 = v55;
      v16 = v54;
      if ( !(_DWORD)v55 )
        goto LABEL_76;
      continue;
    }
    break;
  }
  if ( v22 != 56 )
  {
    if ( (unsigned __int8)(v22 - 71) > 1u )
      goto LABEL_49;
LABEL_58:
    v35 = *(_QWORD *)(v20 + 8);
    if ( !v35 )
      goto LABEL_23;
    while ( 1 )
    {
      v39 = sub_1648700(v35);
      v40 = v49;
      if ( v50 == v49 )
      {
        v41 = &v49[8 * HIDWORD(v51)];
        if ( v49 != (_BYTE *)v41 )
        {
          v42 = 0;
          while ( v39 != (_QWORD *)*v40 )
          {
            if ( *v40 == -2 )
              v42 = v40;
            if ( v41 == ++v40 )
            {
              if ( !v42 )
                goto LABEL_73;
              *v42 = v39;
              --v52;
              ++v48;
              goto LABEL_71;
            }
          }
          goto LABEL_61;
        }
LABEL_73:
        if ( HIDWORD(v51) < (unsigned int)v51 )
          break;
      }
      sub_16CCBA0((__int64)&v48, (__int64)v39);
      if ( v38 )
      {
LABEL_71:
        v43 = (unsigned int)v55;
        if ( (unsigned int)v55 < HIDWORD(v55) )
        {
LABEL_72:
          v44 = (__int64 *)&v54[16 * v43];
          *v44 = v20;
          v44[1] = (__int64)v39;
          LODWORD(v55) = v55 + 1;
          goto LABEL_61;
        }
LABEL_75:
        sub_16CD150((__int64)&v54, v56, 0, 16, v36, v37);
        v43 = (unsigned int)v55;
        goto LABEL_72;
      }
LABEL_61:
      v35 = *(_QWORD *)(v35 + 8);
      if ( !v35 )
        goto LABEL_23;
    }
    ++HIDWORD(v51);
    *v41 = v39;
    v43 = (unsigned int)v55;
    ++v48;
    if ( (unsigned int)v55 < HIDWORD(v55) )
      goto LABEL_72;
    goto LABEL_75;
  }
  if ( (unsigned __int8)sub_15FA290(v20) )
  {
    v24 = **(_QWORD **)(v20 - 24LL * (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(v24 + 8) == 15 )
    {
      v25 = *(_QWORD *)(v24 + 24);
      if ( *(_BYTE *)(v25 + 8) == 13 )
      {
        v26 = v20 + 24 * (1LL - (*(_DWORD *)(v20 + 20) & 0xFFFFFFF));
        v27 = *(_QWORD *)v26;
        if ( *(_DWORD *)(*(_QWORD *)v26 + 32LL) <= 0x40u )
        {
          v28 = *(_QWORD *)(v27 + 24) == 0;
        }
        else
        {
          v45 = *(_DWORD *)(*(_QWORD *)v26 + 32LL);
          v28 = v45 == (unsigned int)sub_16A57B0(v27 + 24);
        }
        if ( v28 )
        {
          v29 = v26 + 24;
          if ( v20 == v29 )
            goto LABEL_58;
          while ( 1 )
          {
            v30 = *(_QWORD *)v29;
            if ( *(_BYTE *)(*(_QWORD *)v29 + 16LL) != 13 )
              goto LABEL_48;
            v31 = *(_DWORD *)(v30 + 32);
            v32 = *(_QWORD **)(v30 + 24);
            v33 = v31 > 0x40 ? v32[(v31 - 1) >> 6] : *(_QWORD *)(v30 + 24);
            if ( (v33 & (1LL << ((unsigned __int8)v31 - 1))) != 0 || *(_DWORD *)(*(_QWORD *)v30 + 8LL) > 0x20FFu )
              goto LABEL_48;
            if ( v31 > 0x40 )
              v32 = (_QWORD *)*v32;
            if ( *(_DWORD *)(v25 + 12) <= (unsigned int)v32 )
              goto LABEL_48;
            v29 += 24;
            v25 = *(_QWORD *)(*(_QWORD *)(v25 + 16) + 8LL * (unsigned int)v32);
            if ( *(_BYTE *)(v25 + 8) != 13 )
              break;
            if ( v29 == v20 )
              goto LABEL_58;
          }
          if ( v29 == v20 )
            goto LABEL_58;
        }
      }
    }
  }
LABEL_48:
  v16 = v54;
LABEL_49:
  if ( v16 != v56 )
    _libc_free((unsigned __int64)v16);
LABEL_51:
  if ( v50 != v49 )
    _libc_free((unsigned __int64)v50);
  return v20;
}

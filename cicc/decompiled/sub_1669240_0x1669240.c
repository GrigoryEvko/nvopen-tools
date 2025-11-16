// Function: sub_1669240
// Address: 0x1669240
//
void __fastcall sub_1669240(__int64 a1, __int64 a2)
{
  unsigned int v2; // edx
  _QWORD *v3; // rcx
  __int64 *v6; // rax
  __int64 *v7; // rsi
  __int64 v8; // r14
  char v9; // dl
  __int64 v10; // rbx
  _QWORD *v11; // r15
  unsigned __int8 v12; // al
  __int64 v13; // r12
  _BYTE *v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdi
  int v17; // ecx
  _QWORD *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // r8
  __int64 v22; // rax
  __int64 *v23; // rsi
  __int64 *v24; // rcx
  __int64 v25; // rdi
  __int64 *v26; // rax
  __int64 v27; // rcx
  _QWORD *v28; // rax
  int v29; // edx
  unsigned __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 *v34; // rdx
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 *v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rdx
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 *v43; // rax
  __int64 v44; // [rsp+0h] [rbp-130h]
  __int64 v45; // [rsp+8h] [rbp-128h]
  __int64 v46; // [rsp+18h] [rbp-118h]
  unsigned __int64 v47[2]; // [rsp+20h] [rbp-110h] BYREF
  char v48; // [rsp+30h] [rbp-100h]
  char v49; // [rsp+31h] [rbp-FFh]
  _QWORD *v50; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v51; // [rsp+48h] [rbp-E8h]
  _QWORD v52[8]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v53; // [rsp+90h] [rbp-A0h] BYREF
  __int64 *v54; // [rsp+98h] [rbp-98h]
  __int64 *v55; // [rsp+A0h] [rbp-90h]
  __int64 v56; // [rsp+A8h] [rbp-88h]
  int v57; // [rsp+B0h] [rbp-80h]
  _BYTE v58[120]; // [rsp+B8h] [rbp-78h] BYREF

  v2 = 1;
  v3 = v52;
  v51 = 0x800000001LL;
  v6 = (__int64 *)v58;
  v50 = v52;
  v53 = 0;
  v54 = (__int64 *)v58;
  v55 = (__int64 *)v58;
  v56 = 8;
  v57 = 0;
  v45 = 0;
  v44 = 0;
  v52[0] = a2;
  v7 = (__int64 *)v58;
  while ( 1 )
  {
    v8 = v3[v2 - 1];
    LODWORD(v51) = v2 - 1;
    if ( v7 != v6 )
      goto LABEL_3;
    v23 = &v6[HIDWORD(v56)];
    if ( v23 != v6 )
    {
      v24 = 0;
      do
      {
        if ( v8 == *v6 )
          goto LABEL_48;
        if ( *v6 == -2 )
          v24 = v6;
        ++v6;
      }
      while ( v23 != v6 );
      if ( v24 )
      {
        *v24 = v8;
        --v57;
        ++v53;
        goto LABEL_4;
      }
    }
    if ( HIDWORD(v56) < (unsigned int)v56 )
    {
      ++HIDWORD(v56);
      *v23 = v8;
      ++v53;
    }
    else
    {
LABEL_3:
      sub_16CCBA0(&v53, v8);
      if ( !v9 )
      {
LABEL_48:
        v49 = 1;
        v47[0] = (unsigned __int64)"FuncletPadInst must not be nested within itself";
        v48 = 3;
        sub_164FF40((__int64 *)a1, (__int64)v47);
        if ( *(_QWORD *)a1 )
          sub_164FA80((__int64 *)a1, v8);
LABEL_12:
        v16 = (unsigned __int64)v55;
        if ( v55 == v54 )
          goto LABEL_14;
LABEL_13:
        _libc_free(v16);
        goto LABEL_14;
      }
    }
LABEL_4:
    v10 = *(_QWORD *)(v8 + 8);
    if ( v10 )
      break;
LABEL_36:
    v2 = v51;
    if ( !(_DWORD)v51 )
      goto LABEL_106;
    v3 = v50;
    v7 = v55;
    v6 = v54;
  }
  v46 = 0;
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = sub_1648700(v10);
        v12 = *((_BYTE *)v11 + 16);
        if ( v12 <= 0x17u )
        {
LABEL_7:
          v13 = *(_QWORD *)a1;
          v49 = 1;
          v47[0] = (unsigned __int64)"Bogus funclet pad use";
          v48 = 3;
          if ( v13 )
          {
            sub_16E2CE0(v47, v13);
            v14 = *(_BYTE **)(v13 + 24);
            if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 16) )
            {
              sub_16E7DE0(v13, 10);
            }
            else
            {
              *(_QWORD *)(v13 + 24) = v14 + 1;
              *v14 = 10;
            }
            v15 = *(_QWORD *)a1;
            *(_BYTE *)(a1 + 72) = 1;
            if ( v15 )
              sub_164FA80((__int64 *)a1, (__int64)v11);
          }
          else
          {
            *(_BYTE *)(a1 + 72) = 1;
          }
          goto LABEL_12;
        }
        if ( v12 == 32 )
        {
          if ( (*((_BYTE *)v11 + 18) & 1) == 0 )
            goto LABEL_53;
          v25 = v11[3 * (1LL - (*((_DWORD *)v11 + 5) & 0xFFFFFFF))];
LABEL_52:
          if ( !v25 )
            goto LABEL_53;
          goto LABEL_63;
        }
        if ( v12 != 34 )
          break;
        if ( (*((_BYTE *)v11 + 18) & 1) == 0 )
          goto LABEL_23;
        if ( (*((_BYTE *)v11 + 23) & 0x40) != 0 )
          v28 = (_QWORD *)*(v11 - 1);
        else
          v28 = &v11[-3 * (*((_DWORD *)v11 + 5) & 0xFFFFFFF)];
        v25 = v28[3];
        if ( !v25 )
        {
LABEL_53:
          v46 = a2;
          v26 = (__int64 *)sub_16498A0(a2);
          v27 = sub_1594470(v26);
LABEL_54:
          if ( v45 )
          {
            if ( v44 != v27 )
            {
              v49 = 1;
              v47[0] = (unsigned __int64)"Unwind edges out of a funclet pad must have the same unwind dest";
              v48 = 3;
              sub_164FF40((__int64 *)a1, (__int64)v47);
              if ( *(_QWORD *)a1 )
              {
                sub_164FA80((__int64 *)a1, a2);
                sub_164FA80((__int64 *)a1, (__int64)v11);
                sub_164FA80((__int64 *)a1, v45);
              }
              goto LABEL_12;
            }
          }
          else
          {
            v45 = (__int64)v11;
            v44 = v27;
            if ( *(_BYTE *)(a2 + 16) == 73 && *(_BYTE *)(v27 + 16) != 16 )
            {
              sub_164ED90(v27);
              v36 = sub_164ED90(a2);
              if ( v37 == v36 )
              {
                v47[0] = a2;
                *(_QWORD *)sub_1668F90(a1 + 760, v47) = v11;
              }
            }
          }
          goto LABEL_56;
        }
LABEL_63:
        v27 = sub_157ED20(v25);
        v29 = *(unsigned __int8 *)(v27 + 16);
        v30 = (unsigned int)(v29 - 34);
        if ( (unsigned int)v30 > 0x36 )
          goto LABEL_23;
        v31 = 0x40018000000001LL;
        if ( !_bittest64(&v31, v30) )
          goto LABEL_23;
        if ( (unsigned __int8)(v29 - 73) <= 1u )
        {
          v32 = *(_QWORD *)(v27 - 24);
LABEL_67:
          if ( v8 == v32 )
            goto LABEL_23;
          goto LABEL_68;
        }
        if ( (*(_BYTE *)(v27 + 23) & 0x40) != 0 )
          v35 = *(__int64 **)(v27 - 8);
        else
          v35 = (__int64 *)(v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF));
        v32 = *v35;
        if ( *v35 )
          goto LABEL_67;
LABEL_68:
        v33 = v8;
        while ( 1 )
        {
          if ( v33 == a2 )
          {
            v46 = a2;
            goto LABEL_54;
          }
          if ( (unsigned __int8)(*(_BYTE *)(v33 + 16) - 73) <= 1u )
          {
            v33 = *(_QWORD *)(v33 - 24);
          }
          else
          {
            v34 = (*(_BYTE *)(v33 + 23) & 0x40) != 0
                ? *(__int64 **)(v33 - 8)
                : (__int64 *)(v33 - 24LL * (*(_DWORD *)(v33 + 20) & 0xFFFFFFF));
            v33 = *v34;
          }
          if ( v32 == v33 )
            break;
          if ( *(_BYTE *)(v33 + 16) == 16 )
            goto LABEL_56;
        }
        v46 = v32;
LABEL_56:
        if ( a2 == v8 )
        {
          v10 = *(_QWORD *)(v10 + 8);
          if ( v10 )
            continue;
        }
        goto LABEL_24;
      }
      if ( v12 == 29 )
      {
        v25 = *(v11 - 3);
        goto LABEL_52;
      }
      if ( v12 != 78 )
        break;
LABEL_23:
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_24;
    }
    if ( v12 != 73 )
    {
      if ( v12 != 33 )
        goto LABEL_7;
      goto LABEL_23;
    }
    v22 = (unsigned int)v51;
    if ( (unsigned int)v51 >= HIDWORD(v51) )
    {
      sub_16CD150(&v50, v52, 0, 8);
      v22 = (unsigned int)v51;
    }
    v50[v22] = v11;
    LODWORD(v51) = v51 + 1;
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v10 );
LABEL_24:
  if ( v46 == v8 || !v46 )
    goto LABEL_36;
  v17 = v51;
  if ( (_DWORD)v51 )
  {
    v18 = &v50[(unsigned int)v51];
    v19 = *(v18 - 1);
    if ( (unsigned __int8)(*(_BYTE *)(v19 + 16) - 73) <= 1u )
    {
LABEL_28:
      v20 = *(_QWORD *)(v19 - 24);
      goto LABEL_30;
    }
    while ( 1 )
    {
      v38 = (*(_BYTE *)(v19 + 23) & 0x40) != 0
          ? *(__int64 **)(v19 - 8)
          : (__int64 *)(v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF));
      v20 = *v38;
LABEL_30:
      while ( v8 != v20 )
      {
        if ( (unsigned __int8)(*(_BYTE *)(v8 + 16) - 73) <= 1u )
        {
          v8 = *(_QWORD *)(v8 - 24);
          if ( v46 == v8 )
            goto LABEL_36;
        }
        else
        {
          if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
            v21 = *(__int64 **)(v8 - 8);
          else
            v21 = (__int64 *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
          v8 = *v21;
          if ( *v21 && v46 == v8 )
            goto LABEL_36;
        }
      }
      --v17;
      --v18;
      LODWORD(v51) = v17;
      if ( !v17 )
        break;
      v19 = *(v18 - 1);
      v8 = v20;
      if ( (unsigned __int8)(*(_BYTE *)(v19 + 16) - 73) <= 1u )
        goto LABEL_28;
    }
  }
LABEL_106:
  if ( v44 )
  {
    v39 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v39 + 16) == 34 )
    {
      if ( (*(_BYTE *)(v39 + 18) & 1) != 0
        && ((*(_BYTE *)(v39 + 23) & 0x40) == 0
          ? (v40 = v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF))
          : (v40 = *(_QWORD *)(v39 - 8)),
            (v41 = *(_QWORD *)(v40 + 24)) != 0) )
      {
        v42 = sub_157ED20(v41);
      }
      else
      {
        v43 = (__int64 *)sub_16498A0(a2);
        v42 = sub_1594470(v43);
      }
      if ( v42 != v44 )
      {
        v49 = 1;
        v47[0] = (unsigned __int64)"Unwind edges out of a catch must have the same unwind dest as the parent catchswitch";
        v48 = 3;
        sub_164FF40((__int64 *)a1, (__int64)v47);
        if ( *(_QWORD *)a1 )
        {
          sub_164FA80((__int64 *)a1, a2);
          sub_164FA80((__int64 *)a1, v45);
          sub_164FA80((__int64 *)a1, v39);
        }
        goto LABEL_12;
      }
    }
  }
  sub_1663F80(a1, a2);
  v16 = (unsigned __int64)v55;
  if ( v55 != v54 )
    goto LABEL_13;
LABEL_14:
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
}

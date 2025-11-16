// Function: sub_1114E60
// Address: 0x1114e60
//
_QWORD *__fastcall sub_1114E60(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rax
  __int64 v5; // r14
  _BYTE *v6; // rax
  bool v7; // bl
  __int64 v8; // rcx
  __int64 v9; // rax
  char v10; // si
  char v11; // di
  _QWORD *v12; // r12
  char v14; // di
  __int64 v15; // rax
  unsigned __int64 v16; // r13
  unsigned __int16 v17; // di
  int v18; // ebx
  _BOOL4 v19; // eax
  int v20; // ebx
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int v24; // eax
  __int64 v25; // rax
  unsigned int v26; // eax
  unsigned int v27; // eax
  __int64 *v28; // r13
  __int64 v29; // r12
  unsigned int v30; // r15d
  bool v31; // cc
  unsigned int v32; // eax
  int v33; // r15d
  __int64 v34; // rbx
  _QWORD **v35; // rdx
  int v36; // ecx
  int v37; // eax
  __int64 *v38; // rax
  __int64 v39; // rsi
  __int64 v40; // rdx
  int v41; // r12d
  __int64 v42; // r12
  __int64 v43; // r13
  __int64 v44; // rdx
  unsigned int v45; // esi
  unsigned int v46; // eax
  __int16 v47; // r13
  __int64 v48; // rax
  unsigned __int8 v49; // r9
  __int64 v50; // r8
  __int64 v51; // rdi
  __int64 v52; // r8
  __int64 v53; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v54; // [rsp+8h] [rbp-C8h]
  __int64 v55; // [rsp+10h] [rbp-C0h]
  __int64 v56; // [rsp+40h] [rbp-90h] BYREF
  __int16 v57; // [rsp+60h] [rbp-70h]
  _BYTE v58[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v59; // [rsp+90h] [rbp-40h]

  v4 = *(_BYTE **)(a2 - 64);
  if ( *v4 != 67
    || (v5 = *((_QWORD *)v4 - 4)) == 0
    || (v6 = *(_BYTE **)(a2 - 32), *v6 != 67)
    || (v55 = *((_QWORD *)v6 - 4)) == 0 )
  {
    v7 = sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F);
    if ( v7 )
    {
      v8 = *(_QWORD *)(a2 - 64);
      v9 = *(_QWORD *)(a2 - 32);
      v14 = *(_BYTE *)v8;
      v10 = *(_BYTE *)v9;
      if ( *(_BYTE *)v8 != 67 || (*(_BYTE *)(v8 + 1) & 4) == 0 )
      {
LABEL_11:
        if ( v10 != 67 )
          return 0;
        v10 = v14;
        goto LABEL_13;
      }
    }
    else
    {
      v8 = *(_QWORD *)(a2 - 64);
      v9 = *(_QWORD *)(a2 - 32);
      v10 = *(_BYTE *)v8;
      v11 = *(_BYTE *)v9;
      if ( *(_BYTE *)v8 == 67
        && (v49 = *(_BYTE *)(v8 + 1), (v49 & 2) != 0)
        && (v5 = *(_QWORD *)(v8 - 32)) != 0
        && (v50 = *(_QWORD *)(v9 + 16)) != 0
        && !*(_QWORD *)(v50 + 8)
        && v11 == 68 )
      {
        v55 = *(_QWORD *)(v9 - 32);
        if ( v55 )
        {
          v54 = sub_B53900(a2) & 0xFFFFFFFFFFLL;
LABEL_64:
          v25 = *(_QWORD *)(a2 - 64);
          goto LABEL_33;
        }
      }
      else
      {
        if ( v11 == 67 )
        {
          if ( (*(_BYTE *)(v9 + 1) & 2) != 0 )
          {
            v5 = *(_QWORD *)(v9 - 32);
            if ( v5 )
            {
              v51 = *(_QWORD *)(v8 + 16);
              if ( v51 )
              {
                if ( !*(_QWORD *)(v51 + 8) && v10 == 68 )
                {
                  v55 = *(_QWORD *)(v8 - 32);
                  if ( v55 )
                  {
                    v54 = sub_B53960(a2) & 0xFFFFFFFFFFLL;
                    goto LABEL_64;
                  }
                  v10 = 68;
                  goto LABEL_13;
                }
              }
            }
          }
          if ( v10 != 67 || (*(_BYTE *)(v8 + 1) & 4) == 0 )
          {
LABEL_13:
            if ( (*(_BYTE *)(v9 + 1) & 4) == 0 )
              return 0;
            v5 = *(_QWORD *)(v9 - 32);
            if ( !v5 )
              return 0;
            v15 = *(_QWORD *)(v8 + 16);
            if ( !v15 || *(_QWORD *)(v15 + 8) || (unsigned __int8)v10 <= 0x1Cu || v10 != 68 && v10 != 69 )
              return 0;
            v55 = *(_QWORD *)(v8 - 32);
            if ( !v55 )
              return 0;
            v54 = sub_B53960(a2) & 0xFFFFFFFFFFLL;
            goto LABEL_78;
          }
          goto LABEL_70;
        }
        if ( v10 != 67 )
          return 0;
        v49 = *(_BYTE *)(v8 + 1);
      }
      if ( ((v49 >> 1) & 2) == 0 )
        return 0;
      v10 = *(_BYTE *)v9;
    }
LABEL_70:
    v5 = *(_QWORD *)(v8 - 32);
    if ( v5 )
    {
      v52 = *(_QWORD *)(v9 + 16);
      v14 = 67;
      if ( v52 && !*(_QWORD *)(v52 + 8) )
      {
        if ( (unsigned __int8)v10 <= 0x1Cu )
          return 0;
        if ( v10 == 68 || v10 == 69 )
        {
          v55 = *(_QWORD *)(v9 - 32);
          if ( !v55 )
            return 0;
          v54 = sub_B53900(a2) & 0xFFFFFFFFFFLL;
LABEL_78:
          v25 = *(_QWORD *)(a2 - 64);
          v7 = *(_BYTE *)v25 == 69 || **(_BYTE **)(a2 - 32) == 69;
          goto LABEL_33;
        }
      }
    }
    else
    {
      v14 = 67;
    }
    goto LABEL_11;
  }
  v16 = sub_B53900(a2) & 0xFFFFFFFFFFLL;
  LOWORD(v54) = v16;
  v17 = *(_WORD *)(a2 + 2) & 0x3F;
  v18 = (*(_BYTE *)(*(_QWORD *)(a2 - 64) + 1LL) >> 1) & 3;
  v19 = (*(_BYTE *)(*(_QWORD *)(a2 - 32) + 1LL) & 2) != 0;
  if ( ((*(_BYTE *)(*(_QWORD *)(a2 - 32) + 1LL) >> 1) & 2) != 0 )
  {
    v20 = (v19 | 2) & v18;
    if ( sub_B532B0(v17) )
    {
      if ( (v20 & 2) != 0 )
        goto LABEL_26;
      return 0;
    }
  }
  else
  {
    v20 = v19 & v18;
    if ( sub_B532B0(v17) )
      return 0;
  }
  if ( !v20 )
    return 0;
LABEL_26:
  v21 = *(_QWORD *)(v5 + 8);
  if ( *(_QWORD *)(v55 + 8) != v21 )
  {
    v22 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 16LL);
    if ( !v22 )
      return 0;
    if ( *(_QWORD *)(v22 + 8) )
      return 0;
    v23 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 16LL);
    if ( !v23 || *(_QWORD *)(v23 + 8) )
      return 0;
  }
  v24 = sub_BCB060(v21);
  if ( !sub_F0C740(a1, v24) )
  {
    v46 = sub_BCB060(*(_QWORD *)(v55 + 8));
    if ( sub_F0C740(a1, v46) )
    {
      v47 = sub_B52F50(v16);
      v48 = v5;
      v5 = v55;
      LOWORD(v54) = v47;
      v55 = v48;
    }
  }
  v25 = *(_QWORD *)(a2 - 64);
  v7 = (v20 & 1) == 0;
LABEL_33:
  v26 = sub_BCB060(*(_QWORD *)(v25 + 8));
  if ( sub_F0C740(a1, v26) )
  {
    v27 = sub_BCB060(*(_QWORD *)(v5 + 8));
    if ( !sub_F0C740(a1, v27) )
      return 0;
  }
  v28 = *(__int64 **)(a1 + 32);
  v57 = 257;
  v29 = *(_QWORD *)(v5 + 8);
  v53 = *(_QWORD *)(v55 + 8);
  v30 = sub_BCB060(v53);
  v31 = v30 <= (unsigned int)sub_BCB060(v29);
  v32 = 38;
  if ( v31 )
    v32 = 39 - (!v7 - 1);
  v33 = v32;
  if ( v29 == v53 )
  {
    v34 = v55;
  }
  else
  {
    v34 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v28[10] + 120LL))(
            v28[10],
            v32,
            v55,
            v29);
    if ( !v34 )
    {
      v59 = 257;
      v34 = sub_B51D30(v33, v55, v29, (__int64)v58, 0, 0);
      if ( (unsigned __int8)sub_920620(v34) )
      {
        v40 = v28[12];
        v41 = *((_DWORD *)v28 + 26);
        if ( v40 )
          sub_B99FD0(v34, 3u, v40);
        sub_B45150(v34, v41);
      }
      (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v28[11] + 16LL))(
        v28[11],
        v34,
        &v56,
        v28[7],
        v28[8]);
      v42 = *v28;
      v43 = *v28 + 16LL * *((unsigned int *)v28 + 2);
      while ( v43 != v42 )
      {
        v44 = *(_QWORD *)(v42 + 8);
        v45 = *(_DWORD *)v42;
        v42 += 16;
        sub_B99FD0(v34, v45, v44);
      }
    }
  }
  v59 = 257;
  v12 = sub_BD2C40(72, unk_3F10FD0);
  if ( v12 )
  {
    v35 = *(_QWORD ***)(v5 + 8);
    v36 = *((unsigned __int8 *)v35 + 8);
    if ( (unsigned int)(v36 - 17) > 1 )
    {
      v39 = sub_BCB2A0(*v35);
    }
    else
    {
      v37 = *((_DWORD *)v35 + 8);
      BYTE4(v56) = (_BYTE)v36 == 18;
      LODWORD(v56) = v37;
      v38 = (__int64 *)sub_BCB2A0(*v35);
      v39 = sub_BCE1B0(v38, v56);
    }
    sub_B523C0((__int64)v12, v39, 53, v54, v5, v34, (__int64)v58, 0, 0, 0);
  }
  return v12;
}

// Function: sub_10A6470
// Address: 0x10a6470
//
unsigned __int8 *__fastcall sub_10A6470(unsigned __int8 *a1)
{
  char v2; // al
  char v3; // bl
  int v4; // eax
  int v5; // eax
  unsigned __int8 *v6; // r13
  __int64 *v8; // rdx
  __int64 v9; // r14
  __int64 v10; // rax
  char v11; // al
  __int64 v12; // r15
  _BYTE *v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rsi
  int v16; // edi
  char v17; // al
  __int64 *v18; // rdx
  _BYTE *v19; // r14
  unsigned __int8 v20; // al
  void *v21; // rax
  _BYTE *v22; // rdx
  char v23; // al
  __int64 *v24; // rdx
  unsigned __int8 *v25; // rdx
  __int64 v26; // r15
  _BYTE *v27; // rsi
  __int64 v28; // rax
  _BYTE *v29; // r15
  _BYTE *v30; // rsi
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // rax
  char v34; // r12
  unsigned int v35; // ebx
  int v36; // esi
  int v37; // ebx
  int v38; // esi
  __int64 v39; // r15
  void **v40; // rax
  _BYTE *v41; // r14
  unsigned int v42; // r15d
  void **v43; // rax
  void **v44; // rcx
  char v45; // al
  _BYTE *v46; // rcx
  char v47; // [rsp+3h] [rbp-6Dh]
  int v48; // [rsp+4h] [rbp-6Ch]
  void **v49; // [rsp+8h] [rbp-68h]
  void **v50; // [rsp+8h] [rbp-68h]
  __int64 *v51[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v52; // [rsp+30h] [rbp-40h]

  v2 = sub_920620((__int64)a1);
  if ( !v2 )
    return 0;
  v3 = v2;
  v4 = *a1;
  if ( (unsigned __int8)v4 > 0x1Cu )
  {
    v5 = v4 - 29;
    if ( v5 != 12 )
      goto LABEL_4;
LABEL_8:
    if ( (a1[7] & 0x40) != 0 )
      v8 = (__int64 *)*((_QWORD *)a1 - 1);
    else
      v8 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v9 = *v8;
    v10 = *(_QWORD *)(*v8 + 16);
    if ( !v10 )
      return 0;
    goto LABEL_11;
  }
  v5 = *((unsigned __int16 *)a1 + 1);
  if ( v5 == 12 )
    goto LABEL_8;
LABEL_4:
  if ( v5 != 16 )
    return 0;
  v17 = a1[7] & 0x40;
  if ( (a1[1] & 0x10) != 0 )
  {
    v18 = v17 ? (__int64 *)*((_QWORD *)a1 - 1) : (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v19 = (_BYTE *)*v18;
    v20 = *(_BYTE *)*v18;
    if ( v20 == 18 )
    {
      v21 = sub_C33340();
      v22 = v19 + 24;
      if ( *((void **)v19 + 3) == v21 )
        v22 = (_BYTE *)*((_QWORD *)v19 + 4);
      v23 = (v22[20] & 7) == 3;
    }
    else
    {
      v39 = *((_QWORD *)v19 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v39 + 8) - 17 > 1 || v20 > 0x15u )
        return 0;
      v40 = (void **)sub_AD7630(*v18, 0, (__int64)v18);
      if ( !v40 || (v49 = v40, *(_BYTE *)v40 != 18) )
      {
        if ( *(_BYTE *)(v39 + 8) == 17 )
        {
          v48 = *(_DWORD *)(v39 + 32);
          if ( v48 )
          {
            v47 = 0;
            v42 = 0;
            while ( 1 )
            {
              v43 = (void **)sub_AD69F0(v19, v42);
              v44 = v43;
              if ( !v43 )
                break;
              v45 = *(_BYTE *)v43;
              v50 = v44;
              if ( v45 != 13 )
              {
                if ( v45 != 18 )
                  return 0;
                v46 = v44[3] == sub_C33340() ? v50[4] : v50 + 3;
                if ( (v46[20] & 7) != 3 )
                  return 0;
                v47 = v3;
              }
              if ( v48 == ++v42 )
              {
                if ( v47 )
                  goto LABEL_33;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v41 = v40 + 3;
      if ( v40[3] == sub_C33340() )
        v41 = v49[4];
      v23 = (v41[20] & 7) == 3;
    }
  }
  else
  {
    v51[0] = 0;
    v24 = v17 ? (__int64 *)*((_QWORD *)a1 - 1) : (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v23 = sub_1008640(v51, *v24);
  }
  if ( !v23 )
    return 0;
LABEL_33:
  if ( (a1[7] & 0x40) != 0 )
    v25 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
  else
    v25 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v9 = *((_QWORD *)v25 + 4);
  v10 = *(_QWORD *)(v9 + 16);
  if ( !v10 )
    return 0;
LABEL_11:
  if ( *(_QWORD *)(v10 + 8) )
    return 0;
  v11 = *(_BYTE *)v9;
  if ( *(_BYTE *)v9 <= 0x1Cu )
    return 0;
  if ( v11 == 47 )
  {
    v26 = *(_QWORD *)(v9 - 64);
    if ( !v26 )
      goto LABEL_15;
    v27 = *(_BYTE **)(v9 - 32);
    if ( *v27 > 0x15u )
      goto LABEL_15;
    v28 = sub_96E680(12, (__int64)v27);
    if ( v28 )
    {
      v52 = 257;
      v14 = v28;
      v15 = v26;
      v16 = 18;
      goto LABEL_21;
    }
    v11 = *(_BYTE *)v9;
  }
  if ( v11 != 50 )
    goto LABEL_15;
  v29 = *(_BYTE **)(v9 - 64);
  if ( v29 )
  {
    v30 = *(_BYTE **)(v9 - 32);
    if ( *v30 <= 0x15u )
    {
      v31 = sub_96E680(12, (__int64)v30);
      if ( v31 )
      {
        v14 = v31;
        v52 = 257;
        v15 = (__int64)v29;
        v16 = 21;
        goto LABEL_21;
      }
      if ( *(_BYTE *)v9 == 50 )
      {
        v29 = *(_BYTE **)(v9 - 64);
        goto LABEL_50;
      }
LABEL_15:
      if ( sub_B451E0((__int64)a1) && *(_BYTE *)v9 == 43 )
      {
        v12 = *(_QWORD *)(v9 - 64);
        if ( v12 )
        {
          v13 = *(_BYTE **)(v9 - 32);
          if ( *v13 <= 0x15u )
          {
            v6 = (unsigned __int8 *)sub_96E680(12, (__int64)v13);
            if ( !v6 )
              return v6;
            v52 = 257;
            v14 = v12;
            v15 = (__int64)v6;
            v16 = 16;
LABEL_21:
            v6 = (unsigned __int8 *)sub_B504D0(v16, v15, v14, (__int64)v51, 0, 0);
            sub_B45260(v6, (__int64)a1, 1);
            return v6;
          }
        }
      }
      return 0;
    }
  }
LABEL_50:
  if ( *v29 > 0x15u )
    return 0;
  v32 = *(_QWORD *)(v9 - 32);
  if ( !v32 )
    return 0;
  v33 = sub_96E680(12, (__int64)v29);
  if ( !v33 )
    goto LABEL_15;
  v52 = 257;
  v6 = (unsigned __int8 *)sub_B504D0(21, v33, v32, (__int64)v51, 0, 0);
  sub_B45260(v6, (__int64)a1, 1);
  v34 = sub_B45210((__int64)a1);
  v35 = sub_B45210(v9);
  v36 = (v35 >> 3) & 1;
  if ( (v34 & 8) == 0 )
    v36 = 0;
  v37 = (v35 >> 2) & 1;
  sub_B450D0((__int64)v6, v36);
  v38 = 0;
  if ( (v34 & 4) != 0 )
    v38 = v37;
  sub_B44F10((__int64)v6, v38);
  return v6;
}

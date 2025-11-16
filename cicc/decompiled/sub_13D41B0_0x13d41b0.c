// Function: sub_13D41B0
// Address: 0x13d41b0
//
unsigned __int64 __fastcall sub_13D41B0(
        unsigned __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        char a5)
{
  unsigned int v10; // edx
  unsigned __int64 result; // rax
  int v12; // eax
  char v13; // al
  _BYTE *v14; // rdi
  _QWORD *v15; // rsi
  char v16; // al
  _BYTE *v17; // rdi
  _BYTE *v18; // rsi
  unsigned int v19; // edx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r8
  bool v22; // cc
  char v23; // al
  char v24; // al
  _BYTE *v25; // rdi
  _BYTE *v26; // rsi
  unsigned int v27; // edx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // r8
  char v30; // al
  char v31; // al
  _BYTE *v32; // rdi
  _QWORD *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // [rsp+0h] [rbp-80h]
  unsigned __int64 v39; // [rsp+0h] [rbp-80h]
  unsigned int v40; // [rsp+8h] [rbp-78h]
  unsigned int v41; // [rsp+8h] [rbp-78h]
  char v42; // [rsp+8h] [rbp-78h]
  unsigned int v43; // [rsp+8h] [rbp-78h]
  char v44; // [rsp+8h] [rbp-78h]
  _BYTE *v45; // [rsp+18h] [rbp-68h] BYREF
  unsigned __int64 v46; // [rsp+20h] [rbp-60h]
  _QWORD *v47; // [rsp+28h] [rbp-58h] BYREF
  unsigned __int64 v48; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v49; // [rsp+38h] [rbp-48h]
  unsigned __int64 v50; // [rsp+40h] [rbp-40h] BYREF
  _QWORD *v51[7]; // [rsp+48h] [rbp-38h] BYREF

  if ( a2 != a3 )
  {
LABEL_2:
    if ( a3 != a1 )
      goto LABEL_3;
    goto LABEL_40;
  }
  v46 = a2;
  v47 = &v45;
  v16 = *(_BYTE *)(a1 + 16);
  if ( v16 == 50 )
  {
    if ( a2 != *(_QWORD *)(a1 - 48) || !(unsigned __int8)sub_13D2630(&v47, *(_BYTE **)(a1 - 24)) )
      goto LABEL_2;
  }
  else
  {
    if ( v16 != 5 || *(_WORD *)(a1 + 18) != 26 || a2 != *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)) )
      goto LABEL_2;
    v17 = *(_BYTE **)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    if ( v17[16] == 13 )
    {
      v18 = v17 + 24;
      v45 = v17 + 24;
      goto LABEL_28;
    }
    if ( *(_BYTE *)(*(_QWORD *)v17 + 8LL) != 16 )
      goto LABEL_2;
    v34 = sub_15A1020(v17);
    if ( !v34 || *(_BYTE *)(v34 + 16) != 13 )
      goto LABEL_2;
    *v47 = v34 + 24;
  }
  v18 = v45;
LABEL_28:
  v19 = *((_DWORD *)v18 + 2);
  v49 = v19;
  if ( v19 <= 0x40 )
  {
    v20 = *(_QWORD *)v18;
LABEL_30:
    v21 = ~v20 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v19);
    v48 = v21;
    goto LABEL_31;
  }
  sub_16A4FD0(&v48, v18);
  v19 = v49;
  if ( v49 <= 0x40 )
  {
    v20 = v48;
    goto LABEL_30;
  }
  sub_16A8F40(&v48);
  v19 = v49;
  v21 = v48;
LABEL_31:
  v22 = *(_DWORD *)(a4 + 8) <= 0x40u;
  LODWORD(v51[0]) = v19;
  v50 = v21;
  v49 = 0;
  if ( v22 )
  {
    v23 = *(_QWORD *)a4 == v21;
  }
  else
  {
    v38 = v21;
    v41 = v19;
    v23 = sub_16A5220(a4, &v50);
    v21 = v38;
    v19 = v41;
  }
  if ( v19 > 0x40 )
  {
    if ( v21 )
    {
      v42 = v23;
      j_j___libc_free_0_0(v21);
      v23 = v42;
      if ( v49 > 0x40 )
      {
        if ( v48 )
        {
          j_j___libc_free_0_0(v48);
          v23 = v42;
        }
      }
    }
  }
  if ( v23 )
  {
LABEL_57:
    if ( !a5 )
      return a1;
    return a2;
  }
  if ( a3 == a1 )
  {
LABEL_40:
    v46 = a3;
    v47 = &v45;
    v24 = *(_BYTE *)(a2 + 16);
    if ( v24 == 50 )
    {
      if ( a3 != *(_QWORD *)(a2 - 48) || !(unsigned __int8)sub_13D2630(&v47, *(_BYTE **)(a2 - 24)) )
        goto LABEL_3;
    }
    else
    {
      if ( v24 != 5 || *(_WORD *)(a2 + 18) != 26 || a3 != *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
        goto LABEL_3;
      v25 = *(_BYTE **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( v25[16] == 13 )
      {
        v26 = v25 + 24;
        v45 = v25 + 24;
        goto LABEL_46;
      }
      if ( *(_BYTE *)(*(_QWORD *)v25 + 8LL) != 16 )
        goto LABEL_3;
      v35 = sub_15A1020(v25);
      if ( !v35 || *(_BYTE *)(v35 + 16) != 13 )
        goto LABEL_3;
      *v47 = v35 + 24;
    }
    v26 = v45;
LABEL_46:
    v27 = *((_DWORD *)v26 + 2);
    v49 = v27;
    if ( v27 > 0x40 )
    {
      sub_16A4FD0(&v48, v26);
      v27 = v49;
      if ( v49 > 0x40 )
      {
        sub_16A8F40(&v48);
        v27 = v49;
        v29 = v48;
LABEL_49:
        v22 = *(_DWORD *)(a4 + 8) <= 0x40u;
        LODWORD(v51[0]) = v27;
        v50 = v29;
        v49 = 0;
        if ( v22 )
        {
          v30 = *(_QWORD *)a4 == v29;
        }
        else
        {
          v39 = v29;
          v43 = v27;
          v30 = sub_16A5220(a4, &v50);
          v29 = v39;
          v27 = v43;
        }
        if ( v27 > 0x40 )
        {
          if ( v29 )
          {
            v44 = v30;
            j_j___libc_free_0_0(v29);
            v30 = v44;
            if ( v49 > 0x40 )
            {
              if ( v48 )
              {
                j_j___libc_free_0_0(v48);
                v30 = v44;
              }
            }
          }
        }
        if ( v30 )
          goto LABEL_57;
        goto LABEL_3;
      }
      v28 = v48;
    }
    else
    {
      v28 = *(_QWORD *)v26;
    }
    v29 = ~v28 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v27);
    v48 = v29;
    goto LABEL_49;
  }
LABEL_3:
  v10 = *(_DWORD *)(a4 + 8);
  if ( v10 > 0x40 )
  {
    v40 = *(_DWORD *)(a4 + 8);
    v12 = sub_16A5940(a4);
    v10 = v40;
    if ( v12 != 1 )
      return 0;
LABEL_6:
    if ( a2 != a3 )
      goto LABEL_7;
    v50 = a2;
    v51[0] = &v45;
    v31 = *(_BYTE *)(a1 + 16);
    if ( v31 == 51 )
    {
      if ( a2 != *(_QWORD *)(a1 - 48) || !(unsigned __int8)sub_13D2630(v51, *(_BYTE **)(a1 - 24)) )
        goto LABEL_7;
    }
    else
    {
      if ( v31 != 5 || *(_WORD *)(a1 + 18) != 27 || a2 != *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)) )
        goto LABEL_7;
      v32 = *(_BYTE **)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
      if ( v32[16] == 13 )
      {
        v33 = v32 + 24;
        v45 = v32 + 24;
        goto LABEL_68;
      }
      if ( *(_BYTE *)(*(_QWORD *)v32 + 8LL) != 16 )
        goto LABEL_7;
      v37 = sub_15A1020(v32);
      if ( !v37 || *(_BYTE *)(v37 + 16) != 13 )
        goto LABEL_7;
      *v51[0] = v37 + 24;
    }
    v33 = v45;
    v10 = *(_DWORD *)(a4 + 8);
LABEL_68:
    if ( v10 <= 0x40 )
    {
      if ( *(_QWORD *)a4 == *v33 )
        goto LABEL_20;
    }
    else if ( (unsigned __int8)sub_16A5220(a4, v33) )
    {
      goto LABEL_20;
    }
LABEL_7:
    if ( a3 != a1 )
      return 0;
    v50 = a1;
    v51[0] = &v45;
    v13 = *(_BYTE *)(a2 + 16);
    if ( v13 == 51 )
    {
      if ( a1 != *(_QWORD *)(a2 - 48) || !(unsigned __int8)sub_13D2630(v51, *(_BYTE **)(a2 - 24)) )
        return 0;
    }
    else
    {
      if ( v13 != 5 || *(_WORD *)(a2 + 18) != 27 || a1 != *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
        return 0;
      v14 = *(_BYTE **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( v14[16] == 13 )
      {
        v15 = v14 + 24;
        v45 = v14 + 24;
        goto LABEL_18;
      }
      if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 16 )
        return 0;
      v36 = sub_15A1020(v14);
      if ( !v36 || *(_BYTE *)(v36 + 16) != 13 )
        return 0;
      *v51[0] = v36 + 24;
    }
    v15 = v45;
LABEL_18:
    if ( *(_DWORD *)(a4 + 8) <= 0x40u )
    {
      if ( *(_QWORD *)a4 != *v15 )
        return 0;
    }
    else if ( !(unsigned __int8)sub_16A5220(a4, v15) )
    {
      return 0;
    }
LABEL_20:
    if ( a5 )
      return a1;
    return a2;
  }
  result = 0;
  if ( *(_QWORD *)a4 && (*(_QWORD *)a4 & (*(_QWORD *)a4 - 1LL)) == 0 )
    goto LABEL_6;
  return result;
}

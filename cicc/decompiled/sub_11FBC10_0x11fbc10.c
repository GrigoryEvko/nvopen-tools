// Function: sub_11FBC10
// Address: 0x11fbc10
//
__int64 __fastcall sub_11FBC10(__int64 a1, char *a2, char a3, char a4)
{
  __int64 v6; // rdi
  __int64 v7; // rdx
  bool v8; // r14
  char v10; // al
  __int64 v11; // r15
  unsigned int v12; // r13d
  bool v13; // al
  __int64 v14; // r13
  unsigned __int8 v15; // al
  unsigned int v16; // r14d
  _BYTE *v17; // rax
  __int64 v18; // r14
  unsigned int v19; // eax
  unsigned int v20; // r13d
  unsigned int v21; // ecx
  __int64 v22; // rdx
  __int64 v23; // r15
  bool v24; // zf
  __int64 v25; // r10
  __int64 v26; // rdi
  int v27; // esi
  int v28; // eax
  unsigned int v29; // edx
  bool v30; // r13
  __int64 v31; // rax
  unsigned int v32; // edx
  unsigned int v33; // r13d
  int v34; // eax
  unsigned int v35; // eax
  __int64 v36; // r13
  _BYTE *v37; // rax
  unsigned int v38; // r13d
  __int64 v39; // r15
  __int64 v40; // rdx
  _BYTE *v41; // rax
  unsigned int v42; // r13d
  bool v43; // al
  bool v44; // r15
  __int64 v45; // rsi
  _BYTE *v46; // rax
  _BYTE *v47; // rdi
  char v48; // al
  unsigned int v49; // r15d
  int v50; // [rsp+0h] [rbp-50h]
  int v51; // [rsp+0h] [rbp-50h]
  unsigned int v52; // [rsp+Ch] [rbp-44h]
  unsigned int v53; // [rsp+Ch] [rbp-44h]
  __int64 v54; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v55; // [rsp+18h] [rbp-38h]

  if ( *a2 == 82 )
  {
    v25 = *((_QWORD *)a2 - 8);
    v26 = *(_QWORD *)(v25 + 8);
    v27 = *(unsigned __int8 *)(v26 + 8);
    if ( (unsigned int)(v27 - 17) <= 1 )
      LOBYTE(v27) = *(_BYTE *)(**(_QWORD **)(v26 + 16) + 8LL);
    if ( (_BYTE)v27 == 12 )
    {
      sub_11FB020(a1, (_BYTE *)v25, *((_QWORD *)a2 - 4), *((_WORD *)a2 + 1) & 0x3F, a3, a4);
      return a1;
    }
    goto LABEL_5;
  }
  v6 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 <= 1 )
    v6 = **(_QWORD **)(v6 + 16);
  v8 = sub_BCAC40(v6, 1);
  if ( !v8 )
    goto LABEL_5;
  v10 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    goto LABEL_5;
  if ( v10 == 67 )
  {
    v18 = *((_QWORD *)a2 - 4);
    if ( v18 )
      goto LABEL_22;
    goto LABEL_5;
  }
  if ( v10 != 59 )
    goto LABEL_5;
  v11 = *((_QWORD *)a2 - 8);
  if ( *(_BYTE *)v11 == 17 )
  {
    v12 = *(_DWORD *)(v11 + 32);
    if ( !v12 )
      goto LABEL_40;
    if ( v12 <= 0x40 )
      v13 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) == *(_QWORD *)(v11 + 24);
    else
      v13 = v12 == (unsigned int)sub_C445E0(v11 + 24);
    goto LABEL_14;
  }
  v36 = *(_QWORD *)(v11 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v36 + 8) - 17 > 1 || *(_BYTE *)v11 > 0x15u )
    goto LABEL_15;
  v37 = sub_AD7630(*((_QWORD *)a2 - 8), 0, v7);
  if ( !v37 || *v37 != 17 )
  {
    if ( *(_BYTE *)(v36 + 8) == 17 )
    {
      v28 = *(_DWORD *)(v36 + 32);
      v29 = 0;
      v30 = 0;
      v50 = v28;
      if ( v28 )
      {
        while ( 1 )
        {
          v52 = v29;
          v31 = sub_AD69F0((unsigned __int8 *)v11, v29);
          v32 = v52;
          if ( !v31 )
            break;
          if ( *(_BYTE *)v31 != 13 )
          {
            if ( *(_BYTE *)v31 != 17 )
              goto LABEL_15;
            v33 = *(_DWORD *)(v31 + 32);
            if ( v33 )
            {
              if ( v33 <= 0x40 )
              {
                v30 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v33) == *(_QWORD *)(v31 + 24);
              }
              else
              {
                v34 = sub_C445E0(v31 + 24);
                v32 = v52;
                v30 = v33 == v34;
              }
              if ( !v30 )
                goto LABEL_15;
            }
            else
            {
              v30 = v8;
            }
          }
          v29 = v32 + 1;
          if ( v50 == v29 )
          {
            if ( !v30 )
              goto LABEL_15;
            goto LABEL_40;
          }
        }
      }
    }
    goto LABEL_15;
  }
  v38 = *((_DWORD *)v37 + 8);
  if ( v38 )
  {
    if ( v38 <= 0x40 )
      v13 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38) == *((_QWORD *)v37 + 3);
    else
      v13 = v38 == (unsigned int)sub_C445E0((__int64)(v37 + 24));
LABEL_14:
    if ( !v13 )
    {
LABEL_15:
      v14 = *((_QWORD *)a2 - 4);
      v15 = *(_BYTE *)v14;
      goto LABEL_16;
    }
  }
LABEL_40:
  v14 = *((_QWORD *)a2 - 4);
  v15 = *(_BYTE *)v14;
  if ( *(_BYTE *)v14 == 67 )
  {
    v18 = *(_QWORD *)(v14 - 32);
    if ( v18 )
      goto LABEL_22;
    goto LABEL_5;
  }
LABEL_16:
  if ( v15 == 17 )
  {
    v16 = *(_DWORD *)(v14 + 32);
    if ( v16 )
    {
      if ( v16 > 0x40 )
      {
        if ( v16 == (unsigned int)sub_C445E0(v14 + 24) )
          goto LABEL_20;
LABEL_5:
        *(_BYTE *)(a1 + 48) = 0;
        return a1;
      }
      v43 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16) == *(_QWORD *)(v14 + 24);
LABEL_60:
      if ( v43 )
        goto LABEL_20;
      goto LABEL_5;
    }
  }
  else
  {
    v39 = *(_QWORD *)(v14 + 8);
    v40 = (unsigned int)*(unsigned __int8 *)(v39 + 8) - 17;
    if ( (unsigned int)v40 > 1 || v15 > 0x15u )
      goto LABEL_5;
    v41 = sub_AD7630(v14, 0, v40);
    if ( !v41 || *v41 != 17 )
    {
      if ( *(_BYTE *)(v39 + 8) == 17 )
      {
        v51 = *(_DWORD *)(v39 + 32);
        if ( v51 )
        {
          v44 = 0;
          v45 = 0;
          while ( 1 )
          {
            v46 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v14, v45);
            v47 = v46;
            if ( !v46 )
              break;
            v48 = *v46;
            if ( v48 != 13 )
            {
              if ( v48 != 17 )
                goto LABEL_5;
              v49 = *((_DWORD *)v47 + 8);
              if ( v49 )
              {
                if ( v49 <= 0x40 )
                  v44 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v49) == *((_QWORD *)v47 + 3);
                else
                  v44 = v49 == (unsigned int)sub_C445E0((__int64)(v47 + 24));
                if ( !v44 )
                  goto LABEL_5;
              }
              else
              {
                v44 = v8;
              }
            }
            v45 = (unsigned int)(v45 + 1);
            if ( v51 == (_DWORD)v45 )
            {
              if ( v44 )
                goto LABEL_20;
              goto LABEL_5;
            }
          }
        }
      }
      goto LABEL_5;
    }
    v42 = *((_DWORD *)v41 + 8);
    if ( v42 )
    {
      if ( v42 <= 0x40 )
      {
        if ( *((_QWORD *)v41 + 3) == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v42) )
          goto LABEL_20;
        goto LABEL_5;
      }
      v43 = v42 == (unsigned int)sub_C445E0((__int64)(v41 + 24));
      goto LABEL_60;
    }
  }
LABEL_20:
  v17 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( *v17 != 67 )
    goto LABEL_5;
  v18 = *((_QWORD *)v17 - 4);
  if ( !v18 )
    goto LABEL_5;
LABEL_22:
  v19 = sub_BCB060(*(_QWORD *)(v18 + 8));
  v55 = v19;
  v20 = v19;
  if ( v19 > 0x40 )
  {
    sub_C43690((__int64)&v54, 1, 0);
    v35 = v55;
    v55 = v20;
    v23 = v54;
    v53 = v35;
    sub_C43690((__int64)&v54, 0, 0);
    v22 = v54;
    v21 = v55;
    v20 = v53;
  }
  else
  {
    v21 = v19;
    v22 = 0;
    v23 = 1;
  }
  v24 = *a2 == 67;
  *(_QWORD *)a1 = v18;
  *(_DWORD *)(a1 + 24) = v20;
  *(_QWORD *)(a1 + 16) = v23;
  *(_DWORD *)(a1 + 8) = v24 + 32;
  *(_DWORD *)(a1 + 40) = v21;
  *(_QWORD *)(a1 + 32) = v22;
  *(_BYTE *)(a1 + 48) = 1;
  return a1;
}

// Function: sub_13D09B0
// Address: 0x13d09b0
//
__int64 __fastcall sub_13D09B0(unsigned __int8 *a1, __int64 a2, char a3, _QWORD *a4)
{
  __int64 v6; // r12
  unsigned __int8 v8; // al
  __int64 v9; // r13
  __int64 v11; // rcx
  char v12; // al
  int v13; // r14d
  _BYTE *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rax
  char v17; // al
  _BYTE *v18; // r14
  bool v19; // al
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // r14
  unsigned int v27; // ecx
  __int64 v28; // rax
  __int64 v29; // rdx
  char v30; // al
  unsigned int v31; // ecx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdx
  unsigned int v35; // r14d
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rcx
  char v39; // al
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // [rsp+8h] [rbp-48h]
  __int64 v43; // [rsp+8h] [rbp-48h]
  unsigned int v44; // [rsp+10h] [rbp-40h]
  __int64 v45; // [rsp+10h] [rbp-40h]
  __int64 v46; // [rsp+18h] [rbp-38h]
  int v47; // [rsp+18h] [rbp-38h]
  int v48; // [rsp+18h] [rbp-38h]

  v6 = a2;
  v8 = a1[16];
  if ( v8 <= 0x10u )
  {
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
    {
      v9 = sub_14D6F90(14, a1, a2, *a4);
      if ( v9 )
        return v9;
      v8 = a1[16];
    }
    if ( v8 == 9 )
      goto LABEL_8;
  }
  if ( *(_BYTE *)(a2 + 16) == 9 )
  {
LABEL_8:
    a2 = 0;
    v9 = sub_15A11D0(*(_QWORD *)a1, 0, 0);
  }
  else
  {
    v9 = sub_13CDA40(a1, (_QWORD *)a2);
  }
  if ( v9 )
    return v9;
  if ( (unsigned __int8)sub_13CBF20(v6) )
    return (__int64)a1;
  if ( !(unsigned __int8)sub_13CC390(v6) )
  {
    if ( !(unsigned __int8)sub_13CC390((__int64)a1) )
      goto LABEL_21;
    goto LABEL_32;
  }
  if ( (a3 & 8) != 0 )
    return (__int64)a1;
  a2 = a4[1];
  if ( (unsigned __int8)sub_14AB3F0(a1, a2, 0) )
    return (__int64)a1;
  if ( (unsigned __int8)sub_13CC390((__int64)a1) )
  {
LABEL_32:
    v17 = *(_BYTE *)(v6 + 16);
    if ( v17 == 38 )
    {
      if ( (unsigned __int8)sub_13CC390(*(_QWORD *)(v6 - 48)) )
      {
        v16 = *(_QWORD *)(v6 - 24);
        if ( v16 )
          return v16;
      }
      goto LABEL_21;
    }
    if ( v17 == 5 && *(_WORD *)(v6 + 18) == 14 )
    {
      v18 = *(_BYTE **)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
      if ( v18[16] == 14 )
      {
        v19 = sub_13D0970((__int64)(v18 + 24));
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v18 + 8LL) != 16 )
          goto LABEL_21;
        v20 = sub_15A1020(*(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
        if ( !v20 || *(_BYTE *)(v20 + 16) != 14 )
        {
          v27 = 0;
          v47 = *(_DWORD *)(*(_QWORD *)v18 + 32LL);
          while ( v47 != v27 )
          {
            a2 = v27;
            v44 = v27;
            v28 = sub_15A0A60(v18, v27);
            v29 = v28;
            if ( !v28 )
              goto LABEL_21;
            v30 = *(_BYTE *)(v28 + 16);
            v31 = v44;
            v42 = v29;
            if ( v30 != 9 )
            {
              if ( v30 != 14 )
                goto LABEL_21;
              v32 = sub_16982C0(v18, a2, v29, v44);
              v31 = v44;
              if ( *(_QWORD *)(v42 + 32) == v32 )
              {
                v34 = *(_QWORD *)(v42 + 40);
                if ( (*(_BYTE *)(v34 + 26) & 7) != 3 )
                  goto LABEL_21;
                v33 = v34 + 8;
              }
              else
              {
                if ( (*(_BYTE *)(v42 + 50) & 7) != 3 )
                  goto LABEL_21;
                v33 = v42 + 32;
              }
              if ( (*(_BYTE *)(v33 + 18) & 8) == 0 )
                goto LABEL_21;
            }
            v27 = v31 + 1;
          }
LABEL_38:
          v16 = *(_QWORD *)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
          if ( v16 )
            return v16;
          goto LABEL_21;
        }
        v19 = sub_13D0970(v20 + 24);
      }
      if ( v19 )
        goto LABEL_38;
    }
LABEL_21:
    if ( (a3 & 8) == 0 || !(unsigned __int8)sub_13CC1F0((__int64)a1) )
      goto LABEL_17;
    v12 = *(_BYTE *)(v6 + 16);
    if ( v12 == 38 )
    {
      if ( !(unsigned __int8)sub_13CC1F0(*(_QWORD *)(v6 - 48)) )
        goto LABEL_17;
      v16 = *(_QWORD *)(v6 - 24);
      if ( !v16 )
        goto LABEL_17;
    }
    else
    {
      if ( v12 != 5 || *(_WORD *)(v6 + 18) != 14 )
        goto LABEL_17;
      v13 = *(_DWORD *)(v6 + 20);
      v14 = *(_BYTE **)(v6 - 24LL * (v13 & 0xFFFFFFF));
      if ( v14[16] == 14 )
      {
        v46 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        if ( *(_QWORD *)(v46 + 32) == sub_16982C0(a1, a2, v14, v11) )
          v15 = *(_QWORD *)(v46 + 40) + 8LL;
        else
          v15 = v46 + 32;
        if ( (*(_BYTE *)(v15 + 18) & 7) != 3 )
          goto LABEL_17;
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 16 )
          goto LABEL_17;
        v21 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
        v22 = sub_15A1020(v21);
        v24 = v21;
        v25 = v22;
        if ( v22 && *(_BYTE *)(v22 + 16) == 14 )
        {
          if ( *(_QWORD *)(v22 + 32) == sub_16982C0(v21, a2, v21, v23) )
            v26 = *(_QWORD *)(v25 + 40) + 8LL;
          else
            v26 = v25 + 32;
          if ( (*(_BYTE *)(v26 + 18) & 7) != 3 )
            goto LABEL_17;
        }
        else
        {
          v35 = 0;
          v48 = *(_DWORD *)(*(_QWORD *)v21 + 32LL);
          while ( v48 != v35 )
          {
            v36 = v24;
            v45 = v24;
            v37 = sub_15A0A60(v24, v35);
            v38 = v37;
            if ( !v37 )
              goto LABEL_17;
            v39 = *(_BYTE *)(v37 + 16);
            v24 = v45;
            v43 = v38;
            if ( v39 != 9 )
            {
              if ( v39 != 14 )
                goto LABEL_17;
              v40 = sub_16982C0(v36, v35, v45, v38);
              v24 = v45;
              v41 = *(_QWORD *)(v43 + 32) == v40 ? *(_QWORD *)(v43 + 40) + 8LL : v43 + 32;
              if ( (*(_BYTE *)(v41 + 18) & 7) != 3 )
                goto LABEL_17;
            }
            ++v35;
          }
        }
        v13 = *(_DWORD *)(v6 + 20);
      }
      v16 = *(_QWORD *)(v6 + 24 * (1LL - (v13 & 0xFFFFFFF)));
      if ( !v16 )
        goto LABEL_17;
    }
    return v16;
  }
LABEL_17:
  if ( (a3 & 2) == 0 || a1 != (unsigned __int8 *)v6 )
    return v9;
  return sub_15A06D0(*(_QWORD *)a1);
}

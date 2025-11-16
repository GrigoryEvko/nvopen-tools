// Function: sub_1006BA0
// Address: 0x1006ba0
//
__int64 __fastcall sub_1006BA0(const __m128i *a1, char *a2, char *a3, char a4)
{
  __int64 v5; // r13
  __int64 v7; // rbx
  char v8; // al
  char v9; // dl
  unsigned __int8 *v10; // r15
  _BYTE *v11; // rsi
  __int64 v13; // rcx
  __int64 v14; // rdx
  int v15; // r11d
  bool v16; // al
  int v17; // r11d
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rax
  bool v22; // bl
  char v23; // al
  bool v24; // al
  bool v25; // al
  __int64 v26; // rax
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // rax
  char v30; // dl
  char *v31; // rcx
  __int64 v32; // rax
  int v33; // edi
  bool v34; // al
  bool v35; // al
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // edi
  __int64 v39; // rax
  __int64 v40; // rax
  int v41; // [rsp+0h] [rbp-70h]
  int v42; // [rsp+0h] [rbp-70h]
  int v43; // [rsp+0h] [rbp-70h]
  __int64 v44; // [rsp+0h] [rbp-70h]
  __int64 v45; // [rsp+0h] [rbp-70h]
  __int64 v46; // [rsp+8h] [rbp-68h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  __int64 v48; // [rsp+18h] [rbp-58h]
  __int64 v49; // [rsp+18h] [rbp-58h]
  __int64 v50; // [rsp+18h] [rbp-58h]
  __int64 v51; // [rsp+18h] [rbp-58h]
  __int64 v52; // [rsp+18h] [rbp-58h]
  __int64 v53; // [rsp+20h] [rbp-50h]
  int v54; // [rsp+20h] [rbp-50h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  bool v57; // [rsp+20h] [rbp-50h]
  int v58; // [rsp+20h] [rbp-50h]
  int v59; // [rsp+20h] [rbp-50h]
  int v60; // [rsp+28h] [rbp-48h]
  __int64 *v61[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = (__int64)a2;
  v7 = (__int64)a3;
  v8 = *a2;
  v9 = *a3;
  if ( (unsigned __int8)(*a2 - 67) <= 0xCu )
  {
    v11 = 0;
    if ( (unsigned __int8)(v9 - 67) > 0xCu )
      return (__int64)v11;
    if ( v8 != v9 )
      return (__int64)v11;
    v31 = *(char **)(v5 - 32);
    v7 = *(_QWORD *)(v7 - 32);
    if ( *(_QWORD *)(v7 + 8) != *((_QWORD *)v31 + 1) )
      return (__int64)v11;
    v10 = (unsigned __int8 *)v5;
    v9 = *(_BYTE *)v7;
    v8 = *v31;
    v5 = *(_QWORD *)(v5 - 32);
  }
  else
  {
    v10 = 0;
  }
  if ( v8 == 82 )
  {
    if ( v9 != 82 )
      return 0;
    if ( a4 )
    {
      v11 = (_BYTE *)sub_1000030(v5, v7, 1u, a1);
      if ( !v11 )
      {
        v11 = (_BYTE *)sub_1000030(v7, v5, 1u, a1);
        if ( !v11 && (*(_QWORD *)(v5 - 64) != *(_QWORD *)(v7 - 64) || (v11 = (_BYTE *)sub_10003F0(v5, v7, 1)) == 0) )
        {
          v11 = (_BYTE *)sub_10008C0(v5, v7, 1);
          if ( !v11 )
          {
            v11 = (_BYTE *)sub_10008C0(v7, v5, 1);
            if ( !v11 )
            {
              v11 = (_BYTE *)sub_1000BA0(v5, v7, a1[4].m128i_i8);
              if ( !v11 )
              {
                v32 = sub_1000BA0(v7, v5, a1[4].m128i_i8);
                v30 = *(_BYTE *)v5;
                v11 = (_BYTE *)v32;
                v23 = *(_BYTE *)v7;
LABEL_72:
                if ( v30 != 83 )
                {
LABEL_73:
                  v5 = (__int64)v11;
                  goto LABEL_50;
                }
                goto LABEL_32;
              }
            }
          }
        }
      }
    }
    else
    {
      v11 = (_BYTE *)sub_1000030(v5, v7, 0, a1);
      if ( !v11 )
      {
        v11 = (_BYTE *)sub_1000030(v7, v5, 0, a1);
        if ( !v11 && (*(_QWORD *)(v5 - 64) != *(_QWORD *)(v7 - 64) || (v11 = (_BYTE *)sub_10003F0(v5, v7, 0)) == 0) )
        {
          v11 = (_BYTE *)sub_10008C0(v5, v7, 0);
          if ( !v11 )
          {
            v11 = (_BYTE *)sub_10008C0(v7, v5, 0);
            if ( !v11 )
            {
              v11 = (_BYTE *)sub_1001020(v5, v7, a1[4].m128i_i8);
              if ( !v11 )
              {
                v29 = sub_1001020(v7, v5, a1[4].m128i_i8);
                v30 = *(_BYTE *)v5;
                v11 = (_BYTE *)v29;
                v23 = *(_BYTE *)v7;
                goto LABEL_72;
              }
            }
          }
        }
      }
    }
    v23 = *(_BYTE *)v7;
    if ( *(_BYTE *)v5 != 83 )
      goto LABEL_52;
LABEL_32:
    if ( v23 != 83 )
      goto LABEL_73;
    goto LABEL_11;
  }
  if ( v8 != 83 || v9 != 83 )
    return 0;
LABEL_11:
  v13 = *(_QWORD *)(v7 - 64);
  v14 = *(_QWORD *)(v5 - 64);
  if ( *(_QWORD *)(v14 + 8) != *(_QWORD *)(v13 + 8) )
    return 0;
  v46 = *(_QWORD *)(v5 - 32);
  v15 = *(_WORD *)(v5 + 2) & 0x3F;
  v47 = *(_QWORD *)(v7 - 32);
  v60 = *(_WORD *)(v7 + 2) & 0x3F;
  if ( (unsigned int)(v15 - 7) > 1 )
    goto LABEL_13;
  v42 = *(_WORD *)(v5 + 2) & 0x3F;
  v49 = *(_QWORD *)(v7 - 64);
  v55 = *(_QWORD *)(v5 - 64);
  v24 = sub_B535B0(v60);
  v14 = v55;
  v13 = v49;
  v15 = v42;
  if ( v24 )
  {
    if ( !a4 )
    {
      v25 = sub_B535C0(v60);
      v14 = v55;
      v13 = v49;
      v15 = v42;
      if ( !v25 )
        goto LABEL_13;
    }
  }
  else
  {
    v35 = sub_B535C0(v60);
    v14 = v55;
    v13 = v49;
    v15 = v42;
    if ( !v35 || a4 )
      goto LABEL_13;
  }
  if ( v14 != v13
    && (*(_BYTE *)v13 != 85
     || (v39 = *(_QWORD *)(v13 - 32)) == 0
     || *(_BYTE *)v39
     || *(_QWORD *)(v39 + 24) != *(_QWORD *)(v13 + 80)
     || *(_DWORD *)(v39 + 36) != 170
     || (v40 = *(_QWORD *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF))) == 0
     || v14 != v40)
    && v14 != v47 )
  {
    if ( *(_BYTE *)v47 != 85 )
      goto LABEL_13;
    v26 = *(_QWORD *)(v47 - 32);
    if ( !v26 )
      goto LABEL_13;
    if ( *(_BYTE *)v26 )
      goto LABEL_13;
    if ( *(_QWORD *)(v26 + 24) != *(_QWORD *)(v47 + 80) )
      goto LABEL_13;
    if ( *(_DWORD *)(v26 + 36) != 170 )
      goto LABEL_13;
    v27 = *(_QWORD *)(v47 - 32LL * (*(_DWORD *)(v47 + 4) & 0x7FFFFFF));
    if ( !v27 || v14 != v27 )
      goto LABEL_13;
  }
  v43 = v15;
  v50 = v13;
  v56 = v14;
  v61[0] = 0;
  v28 = sub_10069D0(v61, v46);
  v14 = v56;
  v13 = v50;
  v15 = v43;
  if ( !v28 )
  {
LABEL_13:
    v48 = v13;
    v53 = v14;
    if ( (unsigned int)(v60 - 7) > 1 )
      return 0;
    v41 = v15;
    v16 = sub_B535B0(v15);
    v17 = v41;
    v18 = v53;
    v19 = v48;
    if ( v16 )
    {
      if ( !a4 )
      {
        v33 = v41;
        v44 = v48;
        v51 = v53;
        v58 = v17;
        v34 = sub_B535C0(v33);
        v17 = v58;
        v18 = v51;
        v19 = v44;
        if ( !v34 )
          return 0;
      }
    }
    else
    {
      v38 = v41;
      v45 = v48;
      v52 = v53;
      v59 = v17;
      if ( !sub_B535C0(v38) )
        return 0;
      v17 = v59;
      v18 = v52;
      v19 = v45;
      if ( a4 )
        return 0;
    }
    if ( v18 != v19
      && (*(_BYTE *)v18 != 85
       || (v36 = *(_QWORD *)(v18 - 32)) == 0
       || *(_BYTE *)v36
       || *(_QWORD *)(v36 + 24) != *(_QWORD *)(v18 + 80)
       || *(_DWORD *)(v36 + 36) != 170
       || (v37 = *(_QWORD *)(v18 - 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF))) == 0
       || v19 != v37)
      && v46 != v19 )
    {
      if ( *(_BYTE *)v46 != 85 )
        return 0;
      v20 = *(_QWORD *)(v46 - 32);
      if ( !v20 )
        return 0;
      if ( *(_BYTE *)v20 )
        return 0;
      if ( *(_QWORD *)(v20 + 24) != *(_QWORD *)(v46 + 80) )
        return 0;
      if ( *(_DWORD *)(v20 + 36) != 170 )
        return 0;
      v21 = *(_QWORD *)(v46 - 32LL * (*(_DWORD *)(v46 + 4) & 0x7FFFFFF));
      if ( !v21 || v19 != v21 )
        return 0;
    }
    v54 = v17;
    v61[0] = 0;
    if ( !(unsigned __int8)sub_10069D0(v61, v47) )
      return 0;
    v22 = sub_B535B0(v54);
    if ( v22 == sub_B535B0(v60) )
      goto LABEL_50;
    goto LABEL_28;
  }
  v57 = sub_B535B0(v43);
  if ( v57 == sub_B535B0(v60) )
  {
    v5 = v7;
    goto LABEL_50;
  }
LABEL_28:
  v5 = sub_AD64A0(*(_QWORD *)(v5 + 8), a4 ^ 1u);
LABEL_50:
  if ( !v5 )
    return 0;
  v11 = (_BYTE *)v5;
LABEL_52:
  if ( !v10 )
    return (__int64)v11;
  if ( *v11 > 0x15u )
    return 0;
  return sub_96F480((unsigned int)*v10 - 29, (__int64)v11, *((_QWORD *)v10 + 1), a1->m128i_i64[0]);
}

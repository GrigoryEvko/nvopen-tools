// Function: sub_1438880
// Address: 0x1438880
//
__int64 __fastcall sub_1438880(__int64 a1)
{
  unsigned __int64 v1; // r13
  _QWORD *v2; // rbx
  char v3; // al
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r12
  int v15; // r12d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  _QWORD *v19; // r12
  __int64 v20; // r15
  unsigned __int8 v21; // al
  unsigned __int64 v22; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rbx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rbx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rbx
  __int64 v56; // rdx
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rbx
  __int64 v61; // rax
  _QWORD v62[7]; // [rsp+8h] [rbp-38h] BYREF

  v1 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = (_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  v3 = *(_BYTE *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  v4 = (a1 >> 2) & 1;
  if ( ((a1 >> 2) & 1) != 0 )
  {
    if ( v3 < 0 )
    {
      v5 = sub_1648A40(a1 & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v5 + v6;
      if ( *(char *)(v1 + 23) >= 0 )
      {
        if ( (unsigned int)(v7 >> 4) )
          goto LABEL_105;
      }
      else if ( (unsigned int)((v7 - sub_1648A40(v1)) >> 4) )
      {
        if ( *(char *)(v1 + 23) < 0 )
        {
          v8 = *(_DWORD *)(sub_1648A40(v1) + 8);
          if ( *(char *)(v1 + 23) < 0 )
          {
            v9 = sub_1648A40(v1);
            v11 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v9 + v10 - 4) - v8);
            goto LABEL_30;
          }
LABEL_106:
          BUG();
        }
LABEL_105:
        BUG();
      }
    }
    v11 = -24;
LABEL_30:
    v19 = (_QWORD *)(v1 + v11);
    if ( v19 == v2 )
      goto LABEL_31;
LABEL_17:
    while ( 1 )
    {
      v20 = *v2;
      v21 = *(_BYTE *)(*v2 + 16LL);
      if ( v21 > 0x10u
        && v21 != 53
        && (v21 != 17
         || !(unsigned __int8)sub_15E0450(*v2)
         && !(unsigned __int8)sub_15E0470(v20)
         && !(unsigned __int8)sub_15E0490(v20)
         && !(unsigned __int8)sub_15E04F0(v20))
        && *(_BYTE *)(*(_QWORD *)v20 + 8LL) == 15 )
      {
        break;
      }
      v2 += 3;
      if ( v2 == v19 )
      {
        if ( !(_BYTE)v4 )
          goto LABEL_35;
LABEL_31:
        if ( !(unsigned __int8)sub_1560260(v1 + 56, 0xFFFFFFFFLL, 36) )
        {
          if ( *(char *)(v1 + 23) < 0 )
          {
            v43 = sub_1648A40(v1);
            v45 = v43 + v44;
            v46 = 0;
            if ( *(char *)(v1 + 23) < 0 )
              v46 = sub_1648A40(v1);
            if ( (unsigned int)((v45 - v46) >> 4) )
              goto LABEL_109;
          }
          v47 = *(_QWORD *)(v1 - 24);
          if ( *(_BYTE *)(v47 + 16)
            || (v62[0] = *(_QWORD *)(v47 + 112), !(unsigned __int8)sub_1560260(v62, 0xFFFFFFFFLL, 36)) )
          {
LABEL_109:
            if ( !(unsigned __int8)sub_1560260(v1 + 56, 0xFFFFFFFFLL, 37) )
            {
              if ( *(char *)(v1 + 23) < 0 )
              {
                v48 = sub_1648A40(v1);
                v50 = v48 + v49;
                v51 = *(char *)(v1 + 23) >= 0 ? 0LL : sub_1648A40(v1);
                if ( v51 != v50 )
                {
                  while ( *(_DWORD *)(*(_QWORD *)v51 + 8LL) <= 1u )
                  {
                    v51 += 16;
                    if ( v50 == v51 )
                      goto LABEL_79;
                  }
                  return 22;
                }
              }
LABEL_79:
              v52 = *(_QWORD *)(v1 - 24);
              if ( *(_BYTE *)(v52 + 16) )
                return 22;
LABEL_80:
              v62[0] = *(_QWORD *)(v52 + 112);
              if ( !(unsigned __int8)sub_1560260(v62, 0xFFFFFFFFLL, 37) )
                return 22;
            }
          }
        }
        return 24;
      }
    }
    v22 = v1 + 56;
    if ( (_BYTE)v4 )
    {
      if ( (unsigned __int8)sub_1560260(v22, 0xFFFFFFFFLL, 36) )
        return 23;
      if ( *(char *)(v1 + 23) >= 0 )
        goto LABEL_110;
      v53 = sub_1648A40(v1);
      v55 = v53 + v54;
      v56 = 0;
      if ( *(char *)(v1 + 23) < 0 )
        v56 = sub_1648A40(v1);
      if ( !(unsigned int)((v55 - v56) >> 4) )
      {
LABEL_110:
        v57 = *(_QWORD *)(v1 - 24);
        if ( !*(_BYTE *)(v57 + 16) )
        {
          v62[0] = *(_QWORD *)(v57 + 112);
          if ( (unsigned __int8)sub_1560260(v62, 0xFFFFFFFFLL, 36) )
            return 23;
        }
      }
      if ( (unsigned __int8)sub_1560260(v1 + 56, 0xFFFFFFFFLL, 37) )
        return 23;
      if ( *(char *)(v1 + 23) < 0 )
      {
        v58 = sub_1648A40(v1);
        v60 = v58 + v59;
        v61 = *(char *)(v1 + 23) >= 0 ? 0LL : sub_1648A40(v1);
        if ( v61 != v60 )
        {
          while ( *(_DWORD *)(*(_QWORD *)v61 + 8LL) <= 1u )
          {
            v61 += 16;
            if ( v60 == v61 )
              goto LABEL_99;
          }
          return 21;
        }
      }
LABEL_99:
      v42 = *(_QWORD *)(v1 - 24);
      if ( *(_BYTE *)(v42 + 16) )
        return 21;
    }
    else
    {
      if ( (unsigned __int8)sub_1560260(v22, 0xFFFFFFFFLL, 36) )
        return 23;
      if ( *(char *)(v1 + 23) >= 0 )
        goto LABEL_111;
      v33 = sub_1648A40(v1);
      v35 = v33 + v34;
      v36 = 0;
      if ( *(char *)(v1 + 23) < 0 )
        v36 = sub_1648A40(v1);
      if ( !(unsigned int)((v35 - v36) >> 4) )
      {
LABEL_111:
        v37 = *(_QWORD *)(v1 - 72);
        if ( !*(_BYTE *)(v37 + 16) )
        {
          v62[0] = *(_QWORD *)(v37 + 112);
          if ( (unsigned __int8)sub_1560260(v62, 0xFFFFFFFFLL, 36) )
            return 23;
        }
      }
      if ( (unsigned __int8)sub_1560260(v1 + 56, 0xFFFFFFFFLL, 37) )
        return 23;
      if ( *(char *)(v1 + 23) < 0 )
      {
        v38 = sub_1648A40(v1);
        v40 = v38 + v39;
        v41 = *(char *)(v1 + 23) >= 0 ? 0LL : sub_1648A40(v1);
        if ( v41 != v40 )
        {
          while ( *(_DWORD *)(*(_QWORD *)v41 + 8LL) <= 1u )
          {
            v41 += 16;
            if ( v40 == v41 )
              goto LABEL_63;
          }
          return 21;
        }
      }
LABEL_63:
      v42 = *(_QWORD *)(v1 - 72);
      if ( *(_BYTE *)(v42 + 16) )
        return 21;
    }
    v62[0] = *(_QWORD *)(v42 + 112);
    if ( !(unsigned __int8)sub_1560260(v62, 0xFFFFFFFFLL, 37) )
      return 21;
    return 23;
  }
  if ( v3 >= 0 )
    goto LABEL_15;
  v12 = sub_1648A40(a1 & 0xFFFFFFFFFFFFFFF8LL);
  v14 = v12 + v13;
  if ( *(char *)(v1 + 23) >= 0 )
  {
    if ( (unsigned int)(v14 >> 4) )
      goto LABEL_105;
    goto LABEL_15;
  }
  if ( !(unsigned int)((v14 - sub_1648A40(v1)) >> 4) )
  {
LABEL_15:
    v18 = -72;
    goto LABEL_16;
  }
  if ( *(char *)(v1 + 23) >= 0 )
    goto LABEL_105;
  v15 = *(_DWORD *)(sub_1648A40(v1) + 8);
  if ( *(char *)(v1 + 23) >= 0 )
    goto LABEL_106;
  v16 = sub_1648A40(v1);
  v18 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v15);
LABEL_16:
  v19 = (_QWORD *)(v1 + v18);
  if ( v19 != v2 )
    goto LABEL_17;
LABEL_35:
  if ( (unsigned __int8)sub_1560260(v1 + 56, 0xFFFFFFFFLL, 36) )
    return 24;
  if ( *(char *)(v1 + 23) >= 0 )
    goto LABEL_112;
  v24 = sub_1648A40(v1);
  v26 = v24 + v25;
  v27 = 0;
  if ( *(char *)(v1 + 23) < 0 )
    v27 = sub_1648A40(v1);
  if ( !(unsigned int)((v26 - v27) >> 4) )
  {
LABEL_112:
    v28 = *(_QWORD *)(v1 - 72);
    if ( !*(_BYTE *)(v28 + 16) )
    {
      v62[0] = *(_QWORD *)(v28 + 112);
      if ( (unsigned __int8)sub_1560260(v62, 0xFFFFFFFFLL, 36) )
        return 24;
    }
  }
  if ( (unsigned __int8)sub_1560260(v1 + 56, 0xFFFFFFFFLL, 37) )
    return 24;
  if ( *(char *)(v1 + 23) >= 0
    || ((v29 = sub_1648A40(v1), v31 = v29 + v30, *(char *)(v1 + 23) >= 0) ? (v32 = 0) : (v32 = sub_1648A40(v1)),
        v32 == v31) )
  {
LABEL_96:
    v52 = *(_QWORD *)(v1 - 72);
    if ( *(_BYTE *)(v52 + 16) )
      return 22;
    goto LABEL_80;
  }
  while ( *(_DWORD *)(*(_QWORD *)v32 + 8LL) <= 1u )
  {
    v32 += 16;
    if ( v31 == v32 )
      goto LABEL_96;
  }
  return 22;
}

// Function: sub_1488C70
// Address: 0x1488c70
//
__int64 __fastcall sub_1488C70(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        char a9)
{
  __int64 v10; // r13
  bool v12; // zf
  __int64 v13; // rbx
  unsigned int v14; // edx
  unsigned int v15; // r8d
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rax
  char v20; // dl
  __int64 v21; // r15
  __int64 v22; // rax
  _QWORD *v23; // rcx
  _QWORD *v24; // rdx
  _QWORD *v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // r14
  __int64 v32; // r14
  __int64 v33; // rax
  char v34; // di
  unsigned int v35; // esi
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rcx
  _QWORD *v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 v51; // rax
  unsigned int v52; // eax
  unsigned int v53; // edx
  unsigned int v54; // edx
  unsigned int v55; // esi
  __int64 v56; // r15
  __int64 v57; // rax
  __int64 v58; // rsi
  __int64 v59; // r13
  unsigned int v60; // edi
  char v61; // r8
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // r14
  __int64 v68; // r14
  __int64 v69; // rax
  char v70; // r8
  unsigned int v71; // edi
  __int64 v72; // rdx
  __int64 v73; // rax
  __int64 v74; // rsi
  __int64 v75; // [rsp+8h] [rbp-98h]
  _QWORD *v76; // [rsp+10h] [rbp-90h]
  unsigned __int8 v77; // [rsp+18h] [rbp-88h]
  unsigned __int8 v78; // [rsp+18h] [rbp-88h]
  _QWORD *v79; // [rsp+18h] [rbp-88h]
  __int64 v80; // [rsp+20h] [rbp-80h]
  __int64 v81; // [rsp+28h] [rbp-78h] BYREF
  __int64 v82; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v83; // [rsp+3Ch] [rbp-64h] BYREF
  unsigned int *v84; // [rsp+40h] [rbp-60h] BYREF
  __int64 v85; // [rsp+48h] [rbp-58h]
  __int64 *v86; // [rsp+50h] [rbp-50h]
  __int64 *v87; // [rsp+58h] [rbp-48h]
  char *v88; // [rsp+60h] [rbp-40h]

  v10 = a4;
  v12 = *(_WORD *)(a3 + 24) == 10;
  v83 = a2;
  v82 = a5;
  v81 = a6;
  if ( !v12 )
  {
    if ( *(_WORD *)(a4 + 24) == 10 )
    {
      v13 = *(_QWORD *)(a4 - 8);
      if ( *(_BYTE *)(v13 + 16) == 77 )
      {
        v75 = a1 + 384;
        sub_1412190(a1 + 384, *(_QWORD *)(a4 - 8));
        v15 = v53;
        if ( !(_BYTE)v53 )
          return v15;
        goto LABEL_7;
      }
    }
    return 0;
  }
  v13 = *(_QWORD *)(a3 - 8);
  if ( *(_BYTE *)(v13 + 16) != 77 )
  {
    if ( *(_WORD *)(a4 + 24) == 10 )
    {
      v80 = *(_QWORD *)(a4 - 8);
      if ( *(_BYTE *)(v80 + 16) == 77 )
      {
        v75 = a1 + 384;
        sub_1412190(a1 + 384, v80);
        v15 = v14;
        if ( !(_BYTE)v14 )
          return v15;
        goto LABEL_6;
      }
    }
    return 0;
  }
  v75 = a1 + 384;
  sub_1412190(a1 + 384, *(_QWORD *)(a3 - 8));
  if ( !v20 )
    return 0;
  if ( *(_WORD *)(v10 + 24) != 10 || (v80 = *(_QWORD *)(v10 - 8), *(_BYTE *)(v80 + 16) != 77) )
  {
    v18 = *(_QWORD *)(v13 + 40);
    if ( *(_WORD *)(v10 + 24) != 7 )
    {
      v85 = a1;
      v84 = &v83;
      v86 = &v82;
      v87 = &v81;
      v88 = &a9;
      v80 = 0;
      goto LABEL_18;
    }
    v85 = a1;
    v84 = &v83;
    v86 = &v82;
    v87 = &v81;
    v88 = &a9;
    v80 = 0;
    goto LABEL_70;
  }
  sub_1412190(v75, v80);
  v23 = *(_QWORD **)(a1 + 400);
  v55 = v54;
  v24 = *(_QWORD **)(a1 + 392);
  v15 = v55;
  v25 = v24;
  if ( (_BYTE)v55 )
  {
    if ( !v13 )
    {
LABEL_6:
      v13 = v80;
LABEL_7:
      v16 = v82;
      v82 = v81;
      v81 = v16;
      v17 = sub_15FF5D0(v83);
      v12 = *(_WORD *)(a3 + 24) == 7;
      v83 = v17;
      v18 = *(_QWORD *)(v13 + 40);
      if ( !v12 )
      {
        v85 = a1;
        v10 = a3;
        v84 = &v83;
        v86 = &v82;
        v87 = &v81;
        v88 = &a9;
        v80 = 0;
        goto LABEL_18;
      }
      v85 = a1;
      v10 = a3;
      v84 = &v83;
      v86 = &v82;
      v87 = &v81;
      v88 = &a9;
      v80 = 0;
      goto LABEL_70;
    }
    v18 = *(_QWORD *)(v13 + 40);
    if ( *(_WORD *)(v10 + 24) == 7 )
    {
      v85 = a1;
      v84 = &v83;
      v86 = &v82;
      v87 = &v81;
      v88 = &a9;
      if ( *(_QWORD *)(v80 + 40) != v18 )
      {
LABEL_70:
        v44 = *(_QWORD *)(v10 + 48);
        if ( **(_QWORD **)(v44 + 32) == v18 )
        {
          if ( (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) == 2 )
          {
            v45 = sub_13FC470(*(_QWORD *)(v10 + 48));
            v46 = sub_1455EB0(v13, v45);
            v47 = sub_146F1B0(a1, v46);
            v15 = sub_1489C20(&v84, v47, **(_QWORD **)(v10 + 32));
            if ( (_BYTE)v15 )
            {
              v48 = sub_13FCB50(v44);
              v49 = sub_1455EB0(v13, v48);
              v50 = sub_146F1B0(a1, v49);
              v51 = sub_1488A90(v10, a1, a7, a8);
              v52 = sub_1489C20(&v84, v50, v51);
              v23 = *(_QWORD **)(a1 + 400);
              v24 = *(_QWORD **)(a1 + 392);
              v15 = v52;
              goto LABEL_22;
            }
            goto LABEL_51;
          }
          goto LABEL_50;
        }
LABEL_18:
        v21 = *(_QWORD *)(v18 + 8);
        if ( v21 )
        {
          while ( 1 )
          {
            v22 = sub_1648700(v21);
            if ( (unsigned __int8)(*(_BYTE *)(v22 + 16) - 25) <= 9u )
              break;
            v21 = *(_QWORD *)(v21 + 8);
            if ( !v21 )
              goto LABEL_21;
          }
LABEL_38:
          v32 = *(_QWORD *)(v22 + 40);
          if ( !sub_146D920(a1, v10, v32) )
            goto LABEL_50;
          v33 = 0x17FFFFFFE8LL;
          v34 = *(_BYTE *)(v13 + 23) & 0x40;
          v35 = *(_DWORD *)(v13 + 20) & 0xFFFFFFF;
          if ( v35 )
          {
            v36 = 24LL * *(unsigned int *)(v13 + 56) + 8;
            v37 = 0;
            do
            {
              v38 = v13 - 24LL * v35;
              if ( v34 )
                v38 = *(_QWORD *)(v13 - 8);
              if ( v32 == *(_QWORD *)(v38 + v36) )
              {
                v33 = 24 * v37;
                goto LABEL_33;
              }
              ++v37;
              v36 += 8;
            }
            while ( v35 != (_DWORD)v37 );
            v33 = 0x17FFFFFFE8LL;
          }
LABEL_33:
          v30 = v34 ? *(_QWORD *)(v13 - 8) : v13 - 24LL * v35;
          v31 = sub_146F1B0(a1, *(_QWORD *)(v30 + v33));
          if ( !(unsigned __int8)sub_1481140(v85, *v84, v31, v10)
            && !(unsigned __int8)sub_1479370(v85, *v84, v31, v10, *v86, *v87)
            && !(unsigned __int8)sub_1489690(v85, *v84, v31, v10, *v86, *v87, *(_DWORD *)v88) )
          {
LABEL_50:
            v15 = 0;
LABEL_51:
            v23 = *(_QWORD **)(a1 + 400);
            v24 = *(_QWORD **)(a1 + 392);
            goto LABEL_22;
          }
          while ( 1 )
          {
            v21 = *(_QWORD *)(v21 + 8);
            if ( !v21 )
              break;
            v22 = sub_1648700(v21);
            if ( (unsigned __int8)(*(_BYTE *)(v22 + 16) - 25) <= 9u )
              goto LABEL_38;
          }
        }
LABEL_21:
        v23 = *(_QWORD **)(a1 + 400);
        v24 = *(_QWORD **)(a1 + 392);
        v15 = 1;
LABEL_22:
        v25 = v24;
        goto LABEL_23;
      }
    }
    else
    {
      v85 = a1;
      v84 = &v83;
      v86 = &v82;
      v87 = &v81;
      v88 = &a9;
      if ( v18 != *(_QWORD *)(v80 + 40) )
        goto LABEL_18;
    }
    v56 = *(_QWORD *)(v18 + 8);
    if ( !v56 )
    {
LABEL_87:
      v15 = 1;
      goto LABEL_22;
    }
    while ( 1 )
    {
      v76 = v23;
      v79 = v24;
      v57 = sub_1648700(v56);
      v24 = v79;
      v23 = v76;
      if ( (unsigned __int8)(*(_BYTE *)(v57 + 16) - 25) <= 9u )
        break;
      v56 = *(_QWORD *)(v56 + 8);
      if ( !v56 )
        goto LABEL_87;
    }
LABEL_102:
    v68 = *(_QWORD *)(v57 + 40);
    v69 = 0x17FFFFFFE8LL;
    v70 = *(_BYTE *)(v13 + 23) & 0x40;
    v71 = *(_DWORD *)(v13 + 20) & 0xFFFFFFF;
    if ( v71 )
    {
      v72 = 24LL * *(unsigned int *)(v13 + 56) + 8;
      v73 = 0;
      do
      {
        v74 = v13 - 24LL * v71;
        if ( v70 )
          v74 = *(_QWORD *)(v13 - 8);
        if ( v68 == *(_QWORD *)(v74 + v72) )
        {
          v69 = 24 * v73;
          goto LABEL_88;
        }
        ++v73;
        v72 += 8;
      }
      while ( v71 != (_DWORD)v73 );
      v69 = 0x17FFFFFFE8LL;
    }
LABEL_88:
    if ( v70 )
      v58 = *(_QWORD *)(v13 - 8);
    else
      v58 = v13 - 24LL * v71;
    v59 = sub_146F1B0(a1, *(_QWORD *)(v58 + v69));
    v60 = *(_DWORD *)(v80 + 20) & 0xFFFFFFF;
    if ( v60 )
    {
      v61 = *(_BYTE *)(v80 + 23) & 0x40;
      v62 = 24LL * *(unsigned int *)(v80 + 56) + 8;
      v63 = 0;
      do
      {
        v64 = v80 - 24LL * v60;
        if ( v61 )
          v64 = *(_QWORD *)(v80 - 8);
        if ( v68 == *(_QWORD *)(v64 + v62) )
        {
          v65 = 24 * v63;
          goto LABEL_97;
        }
        ++v63;
        v62 += 8;
      }
      while ( v60 != (_DWORD)v63 );
      v65 = 0x17FFFFFFE8LL;
    }
    else
    {
      v65 = 0x17FFFFFFE8LL;
      v61 = *(_BYTE *)(v80 + 23) & 0x40;
    }
LABEL_97:
    if ( v61 )
      v66 = *(_QWORD *)(v80 - 8);
    else
      v66 = v80 - 24LL * v60;
    v67 = sub_146F1B0(a1, *(_QWORD *)(v66 + v65));
    if ( (unsigned __int8)sub_1481140(v85, *v84, v59, v67)
      || (unsigned __int8)sub_1479370(v85, *v84, v59, v67, *v86, *v87)
      || (v15 = sub_1489690(v85, *v84, v59, v67, *v86, *v87, *(_DWORD *)v88), (_BYTE)v15) )
    {
      while ( 1 )
      {
        v56 = *(_QWORD *)(v56 + 8);
        if ( !v56 )
          break;
        v57 = sub_1648700(v56);
        if ( (unsigned __int8)(*(_BYTE *)(v57 + 16) - 25) <= 9u )
          goto LABEL_102;
      }
      v23 = *(_QWORD **)(a1 + 400);
      v24 = *(_QWORD **)(a1 + 392);
      v15 = 1;
      goto LABEL_22;
    }
    goto LABEL_51;
  }
  v80 = 0;
LABEL_23:
  if ( v23 == v24 )
  {
    v39 = &v23[*(unsigned int *)(a1 + 412)];
    if ( v39 == v23 )
    {
LABEL_76:
      v25 = &v23[*(unsigned int *)(a1 + 412)];
    }
    else
    {
      while ( v13 != *v25 )
      {
        if ( v39 == ++v25 )
          goto LABEL_76;
      }
    }
  }
  else
  {
    v77 = v15;
    v25 = (_QWORD *)sub_16CC9F0(v75, v13);
    v15 = v77;
    if ( v13 == *v25 )
    {
      v40 = *(_QWORD *)(a1 + 400);
      if ( v40 == *(_QWORD *)(a1 + 392) )
        v41 = *(unsigned int *)(a1 + 412);
      else
        v41 = *(unsigned int *)(a1 + 408);
      v39 = (_QWORD *)(v40 + 8 * v41);
    }
    else
    {
      v26 = *(_QWORD *)(a1 + 400);
      if ( v26 != *(_QWORD *)(a1 + 392) )
        goto LABEL_26;
      v25 = (_QWORD *)(v26 + 8LL * *(unsigned int *)(a1 + 412));
      v39 = v25;
    }
  }
  if ( v25 != v39 )
  {
    *v25 = -2;
    ++*(_DWORD *)(a1 + 416);
  }
LABEL_26:
  if ( v80 )
  {
    v27 = *(_QWORD **)(a1 + 392);
    if ( *(_QWORD **)(a1 + 400) == v27 )
    {
      v29 = &v27[*(unsigned int *)(a1 + 412)];
      if ( v27 == v29 )
      {
LABEL_75:
        v27 = v29;
      }
      else
      {
        while ( v80 != *v27 )
        {
          if ( v29 == ++v27 )
            goto LABEL_75;
        }
      }
    }
    else
    {
      v78 = v15;
      v27 = (_QWORD *)sub_16CC9F0(v75, v80);
      v15 = v78;
      if ( v80 == *v27 )
      {
        v42 = *(_QWORD *)(a1 + 400);
        if ( v42 == *(_QWORD *)(a1 + 392) )
          v43 = *(unsigned int *)(a1 + 412);
        else
          v43 = *(unsigned int *)(a1 + 408);
        v29 = (_QWORD *)(v42 + 8 * v43);
      }
      else
      {
        v28 = *(_QWORD *)(a1 + 400);
        if ( v28 != *(_QWORD *)(a1 + 392) )
          return v15;
        v27 = (_QWORD *)(v28 + 8LL * *(unsigned int *)(a1 + 412));
        v29 = v27;
      }
    }
    if ( v27 != v29 )
    {
      *v27 = -2;
      ++*(_DWORD *)(a1 + 416);
    }
  }
  return v15;
}

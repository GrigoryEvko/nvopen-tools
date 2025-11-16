// Function: sub_1A69110
// Address: 0x1a69110
//
__int64 __fastcall sub_1A69110(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // r13
  int v9; // r12d
  __int64 *v10; // rbx
  __int64 **v11; // r10
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 result; // rax
  unsigned int v15; // edx
  __int64 v16; // r11
  bool v17; // al
  __int64 v18; // rcx
  int v19; // r8d
  _QWORD *v20; // rax
  bool v21; // al
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // rax
  int v25; // edx
  _BYTE *v26; // r9
  __int64 v27; // r8
  _BYTE *v28; // rcx
  __int64 v29; // rdx
  __int64 *v30; // rax
  __int64 v31; // r8
  __int64 v32; // rdx
  __int64 **v33; // r15
  __int64 *v34; // rbx
  __int64 v35; // r12
  int v36; // eax
  unsigned int v37; // edx
  __int64 v38; // rdi
  int v39; // eax
  bool v40; // al
  __int64 v41; // rdx
  unsigned int v42; // ebx
  __int64 *v43; // r12
  unsigned __int64 v44; // r13
  __int64 v45; // rdi
  unsigned int v46; // r14d
  unsigned int v47; // ecx
  __int64 v48; // [rsp+10h] [rbp-90h]
  int v49; // [rsp+18h] [rbp-88h]
  __int64 v50; // [rsp+18h] [rbp-88h]
  __int64 v51; // [rsp+18h] [rbp-88h]
  unsigned int v52; // [rsp+20h] [rbp-80h]
  __int64 v53; // [rsp+20h] [rbp-80h]
  __int64 *v54; // [rsp+20h] [rbp-80h]
  unsigned int v55; // [rsp+20h] [rbp-80h]
  int v56; // [rsp+20h] [rbp-80h]
  int v57; // [rsp+28h] [rbp-78h]
  __int64 v58; // [rsp+28h] [rbp-78h]
  __int64 v59; // [rsp+28h] [rbp-78h]
  __int64 v60; // [rsp+28h] [rbp-78h]
  unsigned __int64 v61; // [rsp+28h] [rbp-78h]
  int v62; // [rsp+28h] [rbp-78h]
  __int64 *v63; // [rsp+28h] [rbp-78h]
  _BYTE *v66; // [rsp+40h] [rbp-60h] BYREF
  __int64 v67; // [rsp+48h] [rbp-58h]
  _BYTE v68[80]; // [rsp+50h] [rbp-50h] BYREF

  v6 = a1 + 192;
  v7 = a6;
  v8 = a4;
  v9 = a2;
  v10 = (__int64 *)a1;
  v11 = *(__int64 ***)(a1 + 184);
  if ( a2 != 1 )
  {
    if ( a2 != 3 )
    {
      if ( *(_DWORD *)(a4 + 32) <= 0x40u )
      {
        if ( *(_QWORD *)(a4 + 24) )
          goto LABEL_11;
      }
      else
      {
        v57 = *(_DWORD *)(a4 + 32);
        if ( (unsigned int)sub_16A57B0(a4 + 24) != v57 )
          goto LABEL_11;
      }
      goto LABEL_5;
    }
    v25 = *(_DWORD *)(a6 + 20);
    v26 = v68;
    v27 = 0;
    v67 = 0x400000000LL;
    v28 = v68;
    v29 = v25 & 0xFFFFFFF;
    v66 = v68;
    v30 = (__int64 *)(v7 + 24 * (1 - v29));
    if ( (__int64 *)v7 != v30 )
    {
      v31 = *v30;
      v32 = 0;
      v33 = v11;
      v34 = v30;
      v35 = *v30;
      while ( 1 )
      {
        *(_QWORD *)&v28[8 * v32] = v35;
        v34 += 3;
        v32 = (unsigned int)(v67 + 1);
        LODWORD(v67) = v67 + 1;
        if ( (__int64 *)v7 == v34 )
          break;
        v35 = *v34;
        if ( HIDWORD(v67) <= (unsigned int)v32 )
        {
          sub_16CD150((__int64)&v66, v68, 0, 8, v31, (int)v26);
          v32 = (unsigned int)v67;
        }
        v28 = v66;
      }
      v27 = (unsigned int)v32;
      v11 = v33;
      v6 = a1 + 192;
      v10 = (__int64 *)a1;
      v9 = a2;
      v28 = v66;
      v29 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    }
    v36 = sub_14A26E0(v11, *(_QWORD *)(v7 + 56), *(__int64 **)(v7 - 24 * v29), (__int64)v28, v27);
    if ( v66 != v68 )
    {
      v62 = v36;
      _libc_free((unsigned __int64)v66);
      v36 = v62;
    }
    if ( !v36 )
      goto LABEL_5;
    v37 = *(_DWORD *)(v8 + 32);
    v38 = v8 + 24;
    if ( v37 <= 0x40 )
    {
      v40 = *(_QWORD *)(v8 + 24) == 1;
    }
    else
    {
      v55 = *(_DWORD *)(v8 + 32);
      v39 = sub_16A57B0(v38);
      v37 = v55;
      v38 = v8 + 24;
      v40 = v55 - 1 == v39;
    }
    if ( !v40 )
    {
      if ( v37 <= 0x40 )
      {
        if ( *(_QWORD *)(v8 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v37) )
          goto LABEL_11;
      }
      else if ( v37 != (unsigned int)sub_16A58F0(v38) )
      {
        goto LABEL_11;
      }
    }
    v41 = v7 + 24 * (1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
    if ( v7 == v41 )
      goto LABEL_5;
    v63 = v10;
    v56 = v9;
    v42 = 0;
    v43 = (__int64 *)(v7 + 24 * (1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)));
    v51 = v8;
    v48 = v7;
    v44 = v41 + 8 * ((unsigned __int64)(v7 - 24 - v41) >> 3) + 24;
    while ( 1 )
    {
      v45 = *v43;
      if ( *(_BYTE *)(*v43 + 16) != 13 )
        goto LABEL_45;
      v46 = *(_DWORD *)(v45 + 32);
      if ( v46 <= 0x40 )
        break;
      if ( v46 != (unsigned int)sub_16A57B0(v45 + 24) )
        goto LABEL_45;
LABEL_46:
      v43 += 3;
      if ( (__int64 *)v44 == v43 )
      {
        v47 = v42;
        v9 = v56;
        v10 = v63;
        v8 = v51;
        v7 = v48;
        if ( v47 <= 1 )
          goto LABEL_5;
LABEL_11:
        v18 = v6;
        v19 = 0;
        if ( v6 != v10[24] )
        {
          while ( 1 )
          {
            v20 = *(_QWORD **)(*(_QWORD *)(v18 + 8) + 48LL);
            if ( (_QWORD *)v7 == v20 || *v20 != *(_QWORD *)v7 )
            {
              v18 = *(_QWORD *)(v18 + 8);
            }
            else
            {
              v53 = v18;
              v49 = v19;
              v60 = *(_QWORD *)(v18 + 8);
              v21 = sub_15CC8F0(v10[21], v20[5], *(_QWORD *)(v7 + 40));
              v19 = v49;
              v18 = *(_QWORD *)(v53 + 8);
              if ( v21 && a3 == *(_QWORD *)(v60 + 24) && a5 == *(_QWORD *)(v60 + 40) && v9 == *(_DWORD *)(v60 + 16) )
              {
                v12 = v18 + 16;
                goto LABEL_6;
              }
            }
            ++v19;
            if ( v10[24] == v18 || v19 == 50 )
              goto LABEL_5;
          }
        }
        goto LABEL_5;
      }
    }
    if ( !*(_QWORD *)(v45 + 24) )
      goto LABEL_46;
LABEL_45:
    ++v42;
    goto LABEL_46;
  }
  v15 = *(_DWORD *)(a4 + 32);
  v16 = a4 + 24;
  if ( v15 > 0x40 )
    goto LABEL_8;
  v50 = a4 + 24;
  v54 = *(__int64 **)(a1 + 184);
  v61 = (__int64)(*(_QWORD *)(a4 + 24) << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
  v22 = sub_1456040(a3);
  v23 = sub_14A2A90(v54, v22, 0, 0, 1u, v61);
  v16 = v50;
  if ( v23 )
    goto LABEL_5;
  v15 = *(_DWORD *)(v8 + 32);
  if ( v15 > 0x40 )
  {
LABEL_8:
    v52 = v15;
    v59 = v16;
    if ( (unsigned int)sub_16A57B0(v16) == v15 - 1 )
      goto LABEL_5;
    v17 = v52 == (unsigned int)sub_16A58F0(v59);
  }
  else
  {
    v24 = *(_QWORD *)(v8 + 24);
    if ( v24 == 1 )
      goto LABEL_5;
    v17 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15) == v24;
  }
  if ( !v17 )
    goto LABEL_11;
LABEL_5:
  v12 = 0;
LABEL_6:
  v58 = v12;
  v13 = sub_22077B0(64);
  *(_DWORD *)(v13 + 16) = v9;
  *(_QWORD *)(v13 + 32) = v8;
  *(_QWORD *)(v13 + 24) = a3;
  *(_QWORD *)(v13 + 48) = v7;
  *(_QWORD *)(v13 + 40) = a5;
  *(_QWORD *)(v13 + 56) = v58;
  result = sub_2208C80(v13, v6);
  ++v10[26];
  return result;
}

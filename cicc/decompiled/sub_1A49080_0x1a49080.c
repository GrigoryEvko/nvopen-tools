// Function: sub_1A49080
// Address: 0x1a49080
//
__int64 __fastcall sub_1A49080(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __int64 a5, char a6)
{
  char v8; // al
  unsigned int v9; // r13d
  unsigned __int8 v11; // r10
  unsigned __int8 v12; // r15
  unsigned __int8 v13; // r11
  __int64 v14; // rsi
  unsigned int v15; // ecx
  unsigned int v16; // r13d
  __int64 v17; // rax
  _QWORD *v19; // rax
  __int64 v20; // r13
  __int64 v21; // rsi
  unsigned int v22; // r13d
  __int64 v23; // rcx
  _QWORD *v24; // rax
  _QWORD *v25; // rax
  unsigned int v26; // edx
  __int64 v27; // rdi
  unsigned int v28; // r8d
  __int64 v29; // rdx
  bool v30; // al
  _QWORD *v31; // rax
  _QWORD *v32; // rax
  bool v33; // al
  int v34; // eax
  unsigned __int64 v35; // rdx
  char v36; // al
  bool v37; // zf
  unsigned int v38; // r13d
  _QWORD *v39; // r15
  bool v40; // al
  unsigned __int8 v41; // [rsp+8h] [rbp-78h]
  unsigned __int8 v43; // [rsp+14h] [rbp-6Ch]
  unsigned int v45; // [rsp+18h] [rbp-68h]
  unsigned __int8 v46; // [rsp+18h] [rbp-68h]
  unsigned __int8 v47; // [rsp+18h] [rbp-68h]
  unsigned __int8 v48; // [rsp+18h] [rbp-68h]
  _QWORD *v49; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v50; // [rsp+28h] [rbp-58h]
  _QWORD *v51; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v52; // [rsp+38h] [rbp-48h]
  _QWORD *v53; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v54; // [rsp+48h] [rbp-38h]

  v8 = *(_BYTE *)(a3 + 16);
  v9 = *(_DWORD *)(*(_QWORD *)a3 + 8LL) >> 8;
  if ( (unsigned __int8)(v8 - 17) > 6u )
  {
    v50 = *(_DWORD *)(*(_QWORD *)a3 + 8LL) >> 8;
    v11 = a4;
    v12 = a5;
    v13 = a6;
    if ( v9 > 0x40 )
    {
      sub_16A4EF0((__int64)&v49, 0, 0);
      v8 = *(_BYTE *)(a3 + 16);
      v11 = a4;
      v13 = a6;
      if ( v8 == 13 )
      {
        v14 = a3 + 24;
        if ( v50 > 0x40 )
        {
LABEL_5:
          sub_16A51C0((__int64)&v49, v14);
          v16 = v50;
          goto LABEL_6;
        }
LABEL_4:
        v15 = *(_DWORD *)(a3 + 32);
        if ( v15 <= 0x40 )
        {
          v50 = *(_DWORD *)(a3 + 32);
          v49 = (_QWORD *)(*(_QWORD *)(a3 + 24) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v15));
          goto LABEL_40;
        }
        goto LABEL_5;
      }
    }
    else
    {
      v49 = 0;
      v14 = a3 + 24;
      if ( v8 == 13 )
        goto LABEL_4;
    }
    if ( (unsigned __int8)(v8 - 35) <= 0x11u )
    {
      if ( (v8 & 0xEF) == 0x23 || v8 == 37 )
      {
        v20 = *(_QWORD *)(a3 - 48);
        v21 = *(_QWORD *)(a3 - 24);
        if ( v8 != 51 )
          goto LABEL_23;
        v41 = v13;
        v43 = v11;
        if ( (unsigned __int8)sub_14BB210(
                                *(_QWORD *)(a3 - 48),
                                v21,
                                *(_QWORD *)(a2 + 232),
                                0,
                                a3,
                                *(_QWORD *)(a2 + 240)) )
        {
          v8 = *(_BYTE *)(a3 + 16);
          v11 = v43;
          v13 = v41;
LABEL_23:
          if ( v8 == 35 )
          {
            if ( v12 != 1 && v13 )
            {
              if ( *(_BYTE *)(v20 + 16) != 13 )
                goto LABEL_100;
              v26 = *(_DWORD *)(v20 + 32);
              v27 = *(_QWORD *)(v20 + 24);
              if ( v26 > 0x40 )
                v27 = *(_QWORD *)(v27 + 8LL * ((v26 - 1) >> 6));
              if ( (v27 & (1LL << ((unsigned __int8)v26 - 1))) != 0 )
              {
LABEL_100:
                if ( *(_BYTE *)(v21 + 16) != 13 )
                  goto LABEL_101;
                v28 = *(_DWORD *)(v21 + 32);
                v29 = *(_QWORD *)(v21 + 24);
                if ( v28 > 0x40 )
                  v29 = *(_QWORD *)(v29 + 8LL * ((v28 - 1) >> 6));
                if ( (v29 & (1LL << ((unsigned __int8)v28 - 1))) != 0 )
                {
LABEL_101:
                  if ( v11 )
                  {
                    v46 = v11;
                    v30 = sub_15F2380(a3);
                    v11 = v46;
                    if ( !v30 )
                      goto LABEL_35;
                  }
                }
              }
              goto LABEL_25;
            }
          }
          else if ( v8 != 37 )
          {
            goto LABEL_25;
          }
          if ( v11 )
          {
            v48 = v11;
            v40 = sub_15F2380(a3);
            v11 = v48;
            if ( !v40 )
              goto LABEL_35;
          }
          if ( v12 )
          {
            v47 = v11;
            v33 = sub_15F2370(a3);
            v11 = v47;
            if ( !v33 )
              goto LABEL_35;
          }
LABEL_25:
          v45 = v11;
          sub_1A49080(&v51, a2, *(_QWORD *)(a3 - 48), v11, v12, 0);
          v22 = v52;
          v23 = v45;
          a5 = v12;
          if ( v52 > 0x40 )
          {
            v34 = sub_16A57B0((__int64)&v51);
            a5 = v12;
            v23 = v45;
            if ( v22 - v34 > 0x40 )
            {
LABEL_28:
              if ( v50 > 0x40 && v49 )
                j_j___libc_free_0_0(v49);
              v16 = v52;
              v49 = v51;
              v50 = v52;
LABEL_6:
              if ( v16 > 0x40 )
              {
                if ( v16 - (unsigned int)sub_16A57B0((__int64)&v49) > 0x40 )
                  goto LABEL_8;
                v19 = (_QWORD *)*v49;
LABEL_17:
                if ( !v19 )
                {
LABEL_11:
                  *(_DWORD *)(a1 + 8) = v50;
                  *(_QWORD *)a1 = v49;
                  return a1;
                }
LABEL_8:
                v17 = *(unsigned int *)(a2 + 8);
                if ( (unsigned int)v17 >= *(_DWORD *)(a2 + 12) )
                {
                  sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, (unsigned __int8)a5, a6);
                  v17 = *(unsigned int *)(a2 + 8);
                }
                *(_QWORD *)(*(_QWORD *)a2 + 8 * v17) = a3;
                ++*(_DWORD *)(a2 + 8);
                goto LABEL_11;
              }
LABEL_40:
              v19 = v49;
              goto LABEL_17;
            }
            v24 = (_QWORD *)*v51;
          }
          else
          {
            v24 = v51;
          }
          if ( v24 )
            goto LABEL_28;
          sub_1A49080(&v53, a2, *(_QWORD *)(a3 - 24), v23, a5, 0);
          if ( v52 > 0x40 && v51 )
            j_j___libc_free_0_0(v51);
          v35 = (unsigned __int64)v53;
          v36 = v54;
          v37 = *(_BYTE *)(a3 + 16) == 37;
          v51 = v53;
          v52 = v54;
          if ( !v37 )
            goto LABEL_28;
          if ( v54 > 0x40 )
          {
            sub_16A4FD0((__int64)&v53, (const void **)&v51);
            v36 = v54;
            if ( v54 > 0x40 )
            {
              sub_16A8F40((__int64 *)&v53);
              goto LABEL_87;
            }
            v35 = (unsigned __int64)v53;
          }
          v53 = (_QWORD *)((0xFFFFFFFFFFFFFFFFLL >> -v36) & ~v35);
LABEL_87:
          sub_16A7400((__int64)&v53);
          v38 = v54;
          v54 = 0;
          v39 = v53;
          if ( v52 > 0x40 && v51 )
          {
            j_j___libc_free_0_0(v51);
            v51 = v39;
            v52 = v38;
            if ( v54 > 0x40 && v53 )
              j_j___libc_free_0_0(v53);
          }
          else
          {
            v51 = v53;
            v52 = v38;
          }
          goto LABEL_28;
        }
      }
LABEL_35:
      v16 = v50;
      goto LABEL_6;
    }
    switch ( v8 )
    {
      case '<':
        if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
          v25 = *(_QWORD **)(a3 - 8);
        else
          v25 = (_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
        sub_1A49080(&v51, a2, *v25, v11, v12, v13);
        sub_16A5A50((__int64)&v53, (__int64 *)&v51, v9);
        if ( v50 > 0x40 )
          goto LABEL_44;
        break;
      case '>':
        if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
          v31 = *(_QWORD **)(a3 - 8);
        else
          v31 = (_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
        sub_1A49080(&v51, a2, *v31, 1, v12, v13);
        sub_16A5B10((__int64)&v53, &v51, v9);
        if ( v50 > 0x40 )
        {
LABEL_44:
          if ( v49 )
            j_j___libc_free_0_0(v49);
        }
        break;
      case '=':
        if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
          v32 = *(_QWORD **)(a3 - 8);
        else
          v32 = (_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
        sub_1A49080(&v51, a2, *v32, 0, 1, 0);
        sub_16A5C50((__int64)&v53, (const void **)&v51, v9);
        if ( v50 <= 0x40 )
          break;
        goto LABEL_44;
      default:
        goto LABEL_35;
    }
    v16 = v54;
    v49 = v53;
    v50 = v54;
    if ( v52 <= 0x40 || !v51 )
      goto LABEL_6;
    j_j___libc_free_0_0(v51);
    goto LABEL_35;
  }
  *(_DWORD *)(a1 + 8) = v9;
  if ( v9 > 0x40 )
    sub_16A4EF0(a1, 0, 0);
  else
    *(_QWORD *)a1 = 0;
  return a1;
}

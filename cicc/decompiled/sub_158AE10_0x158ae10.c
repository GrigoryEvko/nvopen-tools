// Function: sub_158AE10
// Address: 0x158ae10
//
__int64 __fastcall sub_158AE10(__int64 a1, int a2, __int64 a3)
{
  unsigned int v5; // r14d
  unsigned int v6; // eax
  unsigned int v7; // eax
  unsigned int v9; // ebx
  unsigned int v10; // r12d
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rdi
  __int64 v14; // rdi
  unsigned int v15; // ebx
  unsigned int v16; // r12d
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // r15
  __int64 v20; // rdi
  unsigned int v21; // eax
  unsigned int v22; // edx
  __int64 v23; // r9
  bool v24; // cc
  char v25; // al
  unsigned int v26; // ebx
  bool v27; // al
  unsigned int v28; // ebx
  unsigned int v29; // ebx
  __int64 v30; // rax
  unsigned int v31; // ebx
  bool v32; // al
  unsigned int v33; // eax
  unsigned int v34; // r12d
  __int64 v35; // r15
  unsigned int v36; // eax
  unsigned int v37; // r12d
  __int64 v38; // rax
  __int64 v39; // [rsp+0h] [rbp-80h]
  unsigned int v40; // [rsp+8h] [rbp-78h]
  char v41; // [rsp+8h] [rbp-78h]
  __int64 v42; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v43; // [rsp+18h] [rbp-68h]
  __int64 v44; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v45; // [rsp+28h] [rbp-58h]
  __int64 v46; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v47; // [rsp+38h] [rbp-48h]
  __int64 v48; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v49; // [rsp+48h] [rbp-38h]

  if ( sub_158A120(a3) )
  {
    v6 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a1 + 8) = v6;
    if ( v6 > 0x40 )
    {
LABEL_41:
      sub_16A4FD0(a1, a3);
      v21 = *(_DWORD *)(a3 + 24);
      *(_DWORD *)(a1 + 24) = v21;
      if ( v21 <= 0x40 )
        goto LABEL_5;
    }
    else
    {
LABEL_4:
      *(_QWORD *)a1 = *(_QWORD *)a3;
      v7 = *(_DWORD *)(a3 + 24);
      *(_DWORD *)(a1 + 24) = v7;
      if ( v7 <= 0x40 )
      {
LABEL_5:
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
        return a1;
      }
    }
    sub_16A4FD0(a1 + 16, a3 + 16);
    return a1;
  }
  v5 = *(_DWORD *)(a3 + 8);
  switch ( a2 )
  {
    case ' ':
      *(_DWORD *)(a1 + 8) = v5;
      if ( v5 > 0x40 )
        goto LABEL_41;
      goto LABEL_4;
    case '!':
      v47 = *(_DWORD *)(a3 + 8);
      if ( v5 > 0x40 )
        sub_16A4FD0(&v46, a3);
      else
        v46 = *(_QWORD *)a3;
      sub_16A7490(&v46, 1);
      v22 = v47;
      v23 = v46;
      v47 = 0;
      v24 = *(_DWORD *)(a3 + 24) <= 0x40u;
      v49 = v22;
      v48 = v46;
      if ( v24 )
      {
        v25 = *(_QWORD *)(a3 + 16) == v46;
      }
      else
      {
        v39 = v46;
        v40 = v22;
        v25 = sub_16A5220(a3 + 16, &v48);
        v23 = v39;
        v22 = v40;
      }
      if ( v22 > 0x40 )
      {
        if ( v23 )
        {
          v41 = v25;
          j_j___libc_free_0_0(v23);
          v25 = v41;
          if ( v47 > 0x40 )
          {
            if ( v46 )
            {
              j_j___libc_free_0_0(v46);
              v25 = v41;
            }
          }
        }
      }
      if ( !v25 )
      {
        sub_15897D0(a1, v5, 1);
        return a1;
      }
      v49 = *(_DWORD *)(a3 + 8);
      if ( v49 > 0x40 )
        sub_16A4FD0(&v48, a3);
      else
        v48 = *(_QWORD *)a3;
      v47 = *(_DWORD *)(a3 + 24);
      if ( v47 > 0x40 )
        sub_16A4FD0(&v46, a3 + 16);
      else
        v46 = *(_QWORD *)(a3 + 16);
      sub_15898E0(a1, (__int64)&v46, &v48);
      if ( v47 > 0x40 && v46 )
        j_j___libc_free_0_0(v46);
      if ( v49 <= 0x40 )
        return a1;
      v14 = v48;
      if ( !v48 )
        return a1;
      goto LABEL_39;
    case '"':
      sub_158AAD0((__int64)&v42, a3);
      v26 = v43;
      if ( v43 <= 0x40 )
        v27 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v43) == v42;
      else
        v27 = v26 == (unsigned int)sub_16A58F0(&v42);
      if ( v27 )
        goto LABEL_107;
      v49 = v5;
      if ( v5 > 0x40 )
      {
        sub_16A4EF0(&v48, 0, 0);
        v26 = v43;
      }
      else
      {
        v48 = 0;
      }
      goto LABEL_79;
    case '#':
      sub_158AAD0((__int64)&v44, a3);
      v28 = v45;
      if ( v45 <= 0x40 )
      {
        if ( !v44 )
          goto LABEL_101;
      }
      else if ( v28 == (unsigned int)sub_16A57B0(&v44) )
      {
        goto LABEL_101;
      }
      v47 = v5;
      if ( v5 > 0x40 )
      {
        sub_16A4EF0(&v46, 0, 0);
        v28 = v45;
      }
      else
      {
        v46 = 0;
      }
      goto LABEL_94;
    case '$':
      sub_158A9F0((__int64)&v44, a3);
      v29 = v45;
      if ( v45 <= 0x40 )
      {
        v30 = v44;
        if ( !v44 )
          goto LABEL_103;
      }
      else
      {
        if ( v29 == (unsigned int)sub_16A57B0(&v44) )
          goto LABEL_103;
        v30 = v44;
      }
      v49 = v29;
      v48 = v30;
      v45 = 0;
      v47 = v5;
      if ( v5 > 0x40 )
        sub_16A4EF0(&v46, 0, 0);
      else
        v46 = 0;
      goto LABEL_13;
    case '%':
      sub_158A9F0((__int64)&v42, a3);
      v31 = v43;
      if ( v43 <= 0x40 )
        v32 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v43) == v42;
      else
        v32 = v31 == (unsigned int)sub_16A58F0(&v42);
      if ( v32 )
        goto LABEL_105;
      v47 = v31;
      v43 = 0;
      v46 = v42;
      sub_16A7490(&v46, 1);
      v33 = v47;
      v45 = v5;
      v47 = 0;
      v49 = v33;
      v48 = v46;
      if ( v5 > 0x40 )
        sub_16A4EF0(&v44, 0, 0);
      else
        v44 = 0;
      goto LABEL_28;
    case '&':
      sub_158ACE0((__int64)&v42, a3);
      v26 = v43;
      if ( v43 <= 0x40 )
      {
        if ( v42 != (1LL << ((unsigned __int8)v43 - 1)) - 1 )
          goto LABEL_76;
      }
      else if ( (*(_QWORD *)(v42 + 8LL * ((v43 - 1) >> 6)) & (1LL << ((unsigned __int8)v43 - 1))) != 0
             || (v34 = v43 - 1, v34 != (unsigned int)sub_16A58F0(&v42)) )
      {
LABEL_76:
        v49 = v5;
        v35 = 1LL << ((unsigned __int8)v5 - 1);
        if ( v5 > 0x40 )
        {
          sub_16A4EF0(&v48, 0, 0);
          if ( v49 > 0x40 )
          {
            *(_QWORD *)(v48 + 8LL * ((v5 - 1) >> 6)) |= v35;
            v26 = v43;
            goto LABEL_79;
          }
          v26 = v43;
        }
        else
        {
          v48 = 0;
        }
        v48 |= v35;
LABEL_79:
        v45 = v26;
        v44 = v42;
        v43 = 0;
        sub_16A7490(&v44, 1);
        v36 = v45;
        v45 = 0;
        v47 = v36;
        v46 = v44;
        sub_15898E0(a1, (__int64)&v46, &v48);
        if ( v47 > 0x40 && v46 )
          j_j___libc_free_0_0(v46);
        if ( v45 > 0x40 && v44 )
          j_j___libc_free_0_0(v44);
        if ( v49 <= 0x40 )
          goto LABEL_37;
        v20 = v48;
        if ( !v48 )
          goto LABEL_37;
LABEL_36:
        j_j___libc_free_0_0(v20);
        goto LABEL_37;
      }
LABEL_107:
      sub_15897D0(a1, v5, 0);
      goto LABEL_37;
    case '\'':
      sub_158ACE0((__int64)&v44, a3);
      v28 = v45;
      if ( v45 <= 0x40 )
      {
        if ( v44 != 1LL << ((unsigned __int8)v45 - 1) )
          goto LABEL_91;
      }
      else if ( (*(_QWORD *)(v44 + 8LL * ((v45 - 1) >> 6)) & (1LL << ((unsigned __int8)v45 - 1))) == 0
             || (v37 = v45 - 1, v37 != (unsigned int)sub_16A58A0(&v44)) )
      {
LABEL_91:
        v47 = v5;
        v38 = 1LL << ((unsigned __int8)v5 - 1);
        if ( v5 > 0x40 )
        {
          sub_16A4EF0(&v46, 0, 0);
          v38 = 1LL << ((unsigned __int8)v5 - 1);
          if ( v47 > 0x40 )
          {
            *(_QWORD *)(v46 + 8LL * ((v5 - 1) >> 6)) |= 1LL << ((unsigned __int8)v5 - 1);
            v28 = v45;
            goto LABEL_94;
          }
          v28 = v45;
        }
        else
        {
          v46 = 0;
        }
        v46 |= v38;
LABEL_94:
        v49 = v28;
        v48 = v44;
        v45 = 0;
        sub_15898E0(a1, (__int64)&v48, &v46);
        if ( v49 > 0x40 && v48 )
          j_j___libc_free_0_0(v48);
        if ( v47 <= 0x40 )
          goto LABEL_19;
        v13 = v46;
        if ( !v46 )
          goto LABEL_19;
        goto LABEL_18;
      }
LABEL_101:
      sub_15897D0(a1, v5, 1);
      goto LABEL_19;
    case '(':
      sub_158ABC0((__int64)&v44, a3);
      v9 = v45;
      if ( v45 <= 0x40 )
      {
        v11 = v44;
        if ( v44 != 1LL << ((unsigned __int8)v45 - 1) )
          goto LABEL_10;
      }
      else
      {
        v11 = v44;
        if ( (*(_QWORD *)(v44 + 8LL * ((v45 - 1) >> 6)) & (1LL << ((unsigned __int8)v45 - 1))) == 0
          || (v11 = v44, v10 = v45 - 1, v10 != (unsigned int)sub_16A58A0(&v44)) )
        {
LABEL_10:
          v49 = v9;
          v48 = v11;
          v45 = 0;
          v47 = v5;
          v12 = 1LL << ((unsigned __int8)v5 - 1);
          if ( v5 > 0x40 )
          {
            sub_16A4EF0(&v46, 0, 0);
            if ( v47 > 0x40 )
            {
              *(_QWORD *)(v46 + 8LL * ((v5 - 1) >> 6)) |= v12;
              goto LABEL_13;
            }
          }
          else
          {
            v46 = 0;
          }
          v46 |= v12;
LABEL_13:
          sub_15898E0(a1, (__int64)&v46, &v48);
          if ( v47 > 0x40 && v46 )
            j_j___libc_free_0_0(v46);
          if ( v49 <= 0x40 || (v13 = v48) == 0 )
          {
LABEL_19:
            if ( v45 > 0x40 )
            {
              v14 = v44;
              if ( v44 )
                goto LABEL_39;
            }
            return a1;
          }
LABEL_18:
          j_j___libc_free_0_0(v13);
          goto LABEL_19;
        }
      }
LABEL_103:
      sub_15897D0(a1, v5, 0);
      goto LABEL_19;
    case ')':
      sub_158ABC0((__int64)&v42, a3);
      v15 = v43;
      if ( v43 <= 0x40 )
      {
        v17 = v42;
        if ( v42 != (1LL << ((unsigned __int8)v43 - 1)) - 1 )
          goto LABEL_25;
LABEL_105:
        sub_15897D0(a1, v5, 1);
        goto LABEL_37;
      }
      v17 = v42;
      if ( (*(_QWORD *)(v42 + 8LL * ((v43 - 1) >> 6)) & (1LL << ((unsigned __int8)v43 - 1))) == 0 )
      {
        v17 = v42;
        v16 = v43 - 1;
        if ( v16 == (unsigned int)sub_16A58F0(&v42) )
          goto LABEL_105;
      }
LABEL_25:
      v46 = v17;
      v47 = v15;
      v43 = 0;
      sub_16A7490(&v46, 1);
      v18 = v47;
      v45 = v5;
      v47 = 0;
      v49 = v18;
      v48 = v46;
      v19 = 1LL << ((unsigned __int8)v5 - 1);
      if ( v5 > 0x40 )
      {
        sub_16A4EF0(&v44, 0, 0);
        if ( v45 > 0x40 )
        {
          *(_QWORD *)(v44 + 8LL * ((v5 - 1) >> 6)) |= v19;
          goto LABEL_28;
        }
      }
      else
      {
        v44 = 0;
      }
      v44 |= v19;
LABEL_28:
      sub_15898E0(a1, (__int64)&v44, &v48);
      if ( v45 > 0x40 && v44 )
        j_j___libc_free_0_0(v44);
      if ( v49 > 0x40 && v48 )
        j_j___libc_free_0_0(v48);
      if ( v47 > 0x40 )
      {
        v20 = v46;
        if ( v46 )
          goto LABEL_36;
      }
LABEL_37:
      if ( v43 > 0x40 )
      {
        v14 = v42;
        if ( v42 )
LABEL_39:
          j_j___libc_free_0_0(v14);
      }
      return a1;
  }
}

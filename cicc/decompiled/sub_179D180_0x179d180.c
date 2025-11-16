// Function: sub_179D180
// Address: 0x179d180
//
__int64 __fastcall sub_179D180(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5)
{
  unsigned __int8 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 *v8; // r15
  unsigned int v9; // eax
  unsigned int v10; // r12d
  _QWORD *v11; // rax
  __int64 v12; // rax
  unsigned int v13; // r13d
  __int64 v15; // r12
  unsigned int v16; // r13d
  __int64 v17; // rax
  _QWORD *v18; // r9
  unsigned int v19; // r13d
  _QWORD *v20; // r12
  __int64 v21; // rax
  _BYTE *v22; // rdi
  unsigned __int8 v23; // al
  __int64 v24; // rdi
  _QWORD *v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // r8
  unsigned int v28; // ecx
  unsigned int v29; // r12d
  unsigned __int64 v30; // rdx
  unsigned int v31; // eax
  unsigned int v32; // edx
  __int64 v33; // rsi
  __int64 *v34; // rbx
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // [rsp+0h] [rbp-70h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+10h] [rbp-60h]
  _QWORD *v40; // [rsp+18h] [rbp-58h]
  unsigned int v41; // [rsp+18h] [rbp-58h]
  unsigned __int64 v42; // [rsp+18h] [rbp-58h]
  __int64 v43; // [rsp+18h] [rbp-58h]
  char v44; // [rsp+18h] [rbp-58h]
  __int64 v45; // [rsp+18h] [rbp-58h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  unsigned __int64 v47; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v48; // [rsp+28h] [rbp-48h]
  unsigned __int64 v49; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v50; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    while ( 1 )
    {
      v5 = *(_BYTE *)(a1 + 16);
      if ( v5 <= 0x10u )
        return 1;
      if ( v5 <= 0x17u )
        return 0;
      v6 = *(_QWORD *)(a1 + 8);
      v7 = a1;
      if ( !v6 || *(_QWORD *)(v6 + 8) )
        return 0;
      v8 = a4;
      v9 = v5 - 24;
      if ( v5 == 77 )
      {
        v17 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        {
          v18 = *(_QWORD **)(a1 - 8);
          v40 = &v18[v17];
        }
        else
        {
          v40 = (_QWORD *)a1;
          v18 = (_QWORD *)(a1 - v17 * 8);
        }
        if ( v18 != v40 )
        {
          v19 = (unsigned __int8)a3;
          v20 = v18;
          while ( (unsigned __int8)sub_179D180(*v20, (unsigned int)a2, v19, v8, a1) )
          {
            v20 += 3;
            if ( v40 == v20 )
              return 1;
          }
          return 0;
        }
        return 1;
      }
      if ( v9 <= 0x35 )
        break;
      if ( v5 != 79 )
        return 0;
      v15 = *(_QWORD *)(a1 - 24);
      v16 = (unsigned __int8)a3;
      if ( !(unsigned __int8)sub_179D180(*(_QWORD *)(a1 - 48), a2, (unsigned __int8)a3, a4, a1) )
        return 0;
      a5 = a1;
      a4 = v8;
      a3 = v16;
      a2 = (unsigned int)a2;
      a1 = v15;
    }
    if ( v9 <= 0x18 )
      break;
    if ( (unsigned int)v5 - 50 > 2 )
      return 0;
    v10 = (unsigned __int8)a3;
    v11 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
        ? *(_QWORD **)(a1 - 8)
        : (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( !(unsigned __int8)sub_179D180(*v11, (unsigned int)a2, (unsigned __int8)a3, a4, a1) )
      return 0;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v12 = *(_QWORD *)(a1 - 8);
    else
      v12 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    a1 = *(_QWORD *)(v12 + 24);
    a5 = v7;
    a4 = v8;
    a3 = v10;
    a2 = (unsigned int)a2;
  }
  if ( v9 <= 0x16 )
    return 0;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v21 = *(_QWORD *)(a1 - 8);
  else
    v21 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v22 = *(_BYTE **)(v21 + 24);
  v23 = v22[16];
  if ( v23 == 13 )
  {
    v24 = (__int64)(v22 + 24);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v22 + 8LL) != 16 )
      return 0;
    v39 = a5;
    v44 = a3;
    if ( v23 > 0x10u )
      return 0;
    v35 = sub_15A1020(v22, a2, a3, *(_QWORD *)v22);
    if ( !v35 || *(_BYTE *)(v35 + 16) != 13 )
      return 0;
    v5 = *(_BYTE *)(v7 + 16);
    LOBYTE(a3) = v44;
    v24 = v35 + 24;
    a5 = v39;
  }
  v13 = 1;
  if ( (_BYTE)a3 != (v5 == 47) )
  {
    v41 = *(_DWORD *)(v24 + 8);
    if ( v41 <= 0x40 )
    {
      v25 = *(_QWORD **)v24;
      if ( (unsigned int)a2 == *(_QWORD *)v24 )
        return v13;
      if ( (unsigned __int64)(unsigned int)a2 >= *(_QWORD *)v24 )
        return 0;
LABEL_36:
      v38 = a5;
      v42 = (unsigned __int64)v25;
      v26 = sub_16431D0(*(_QWORD *)v7);
      v27 = v38;
      if ( v42 >= v26 )
        return 0;
      v50 = v26;
      v28 = v26 - v42;
      if ( v5 != 47 )
        v28 = v42 - a2;
      v29 = v28;
      if ( v26 > 0x40 )
      {
        sub_16A4EF0((__int64)&v49, 0, 0);
        v27 = v38;
      }
      else
      {
        v49 = 0;
      }
      if ( !(_DWORD)a2 )
        goto LABEL_73;
      if ( (unsigned int)a2 > 0x40 )
      {
        v46 = v27;
        sub_16A5260(&v49, 0, a2);
        v31 = v50;
        v27 = v46;
      }
      else
      {
        v30 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a2);
        v31 = v50;
        if ( v50 <= 0x40 )
        {
          v48 = v50;
          v49 |= v30;
          goto LABEL_45;
        }
        *(_QWORD *)v49 |= v30;
LABEL_73:
        v31 = v50;
      }
      v48 = v31;
      if ( v31 <= 0x40 )
      {
LABEL_45:
        v47 = v49;
        v32 = v31;
      }
      else
      {
        v45 = v27;
        sub_16A4FD0((__int64)&v47, (const void **)&v49);
        v31 = v48;
        v27 = v45;
        if ( v48 > 0x40 )
        {
          sub_16A7DC0((__int64 *)&v47, v29);
          v32 = v50;
          v27 = v45;
LABEL_49:
          if ( v32 > 0x40 && v49 )
          {
            v43 = v27;
            j_j___libc_free_0_0(v49);
            v27 = v43;
          }
          if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
            v34 = *(__int64 **)(v7 - 8);
          else
            v34 = (__int64 *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
          v13 = sub_14C1670(*v34, (__int64)&v47, v8[333], 0, v8[330], v27, v8[332]);
          if ( v48 > 0x40 && v47 )
            j_j___libc_free_0_0(v47);
          return v13;
        }
        v32 = v50;
      }
      v33 = 0;
      if ( v29 != v31 )
        v33 = v47 << v29;
      v47 = v33 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v31);
      goto LABEL_49;
    }
    v37 = a5;
    v36 = sub_16A57B0(v24);
    a5 = v37;
    if ( v41 - v36 > 0x40 )
      return 0;
    v25 = **(_QWORD ***)v24;
    if ( (_QWORD *)(unsigned int)a2 != v25 )
    {
      if ( (unsigned int)a2 >= (unsigned __int64)v25 )
        return 0;
      goto LABEL_36;
    }
  }
  return v13;
}

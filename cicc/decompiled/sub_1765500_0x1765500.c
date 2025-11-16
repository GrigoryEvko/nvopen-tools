// Function: sub_1765500
// Address: 0x1765500
//
__int64 __fastcall sub_1765500(__int64 *a1, __int64 a2, _QWORD **a3)
{
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v6; // rcx
  int v7; // eax
  unsigned int v9; // eax
  __int64 **v10; // r14
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx
  _QWORD **v19; // rcx
  int v20; // r13d
  _QWORD **v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 **v25; // rdi
  __int64 v26; // rsi
  unsigned int v27; // ecx
  int v28; // eax
  unsigned __int64 v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned __int64 v33; // rdx
  __int64 v34; // rdx
  int v35; // eax
  __int64 v36; // rcx
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rax
  _QWORD **v40; // [rsp+0h] [rbp-50h]
  unsigned int v42; // [rsp+8h] [rbp-48h]
  __int64 v44; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v45; // [rsp+18h] [rbp-38h]

  v3 = 0;
  v4 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v4 + 16) != 78 )
    return v3;
  v6 = *(_QWORD *)(v4 - 24);
  if ( *(_BYTE *)(v6 + 16) )
    return v3;
  if ( (*(_BYTE *)(v6 + 33) & 0x20) == 0 )
    return v3;
  v7 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v7) &= ~0x80u;
  if ( (unsigned int)(v7 - 32) > 1 )
    return v3;
  v9 = *(_DWORD *)(v6 + 36);
  v10 = *(__int64 ***)v4;
  if ( v9 == 32 )
  {
    v27 = *((_DWORD *)a3 + 2);
    if ( v27 <= 0x40 )
    {
      v30 = *a3;
      if ( *a3 )
      {
        v29 = v27;
LABEL_40:
        v3 = 0;
        if ( v30 != (_QWORD *)v29 )
          return v3;
LABEL_41:
        v22 = v4;
        sub_170B990(*a1, v4);
        v31 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
        v23 = 4 * v31;
        v32 = *(_QWORD *)(v4 - 24 * v31);
        if ( v32 )
        {
          if ( *(_QWORD *)(a2 - 48) )
          {
            v24 = *(_QWORD *)(a2 - 40);
            v33 = *(_QWORD *)(a2 - 32) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v33 = v24;
            if ( v24 )
              *(_QWORD *)(v24 + 16) = *(_QWORD *)(v24 + 16) & 3LL | v33;
          }
          *(_QWORD *)(a2 - 48) = v32;
          v34 = *(_QWORD *)(v32 + 8);
          v22 = v32 + 8;
          *(_QWORD *)(a2 - 40) = v34;
          if ( v34 )
          {
            v24 = (a2 - 40) | *(_QWORD *)(v34 + 16) & 3LL;
            *(_QWORD *)(v34 + 16) = v24;
          }
          *(_QWORD *)(a2 - 32) = v22 | *(_QWORD *)(a2 - 32) & 3LL;
          v23 = a2 - 48;
          *(_QWORD *)(v32 + 8) = a2 - 48;
        }
        else if ( *(_QWORD *)(a2 - 48) )
        {
          v23 = *(_QWORD *)(a2 - 40);
          v38 = *(_QWORD *)(a2 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v38 = v23;
          if ( v23 )
          {
            v24 = *(_QWORD *)(v23 + 16) & 3LL;
            *(_QWORD *)(v23 + 16) = v24 | v38;
          }
          *(_QWORD *)(a2 - 48) = 0;
        }
        v25 = v10;
        if ( !(_BYTE)v3 )
        {
          v26 = sub_15A04A0(v10);
          goto LABEL_33;
        }
LABEL_32:
        v26 = sub_15A06D0(v25, v22, v23, v24);
LABEL_33:
        v3 = a2;
        sub_1593B40((_QWORD *)(a2 - 24), v26);
        return v3;
      }
    }
    else
    {
      v42 = *((_DWORD *)a3 + 2);
      v28 = sub_16A57B0((__int64)a3);
      if ( v42 != v28 )
      {
        v29 = v42;
        if ( v42 - v28 > 0x40 )
          return v3;
        v30 = (_QWORD *)**a3;
        goto LABEL_40;
      }
    }
    LOBYTE(v3) = 1;
    goto LABEL_41;
  }
  if ( v9 > 0x20 )
  {
    if ( v9 != 33 )
      return v3;
  }
  else
  {
    if ( v9 == 6 )
    {
      sub_170B990(*a1, *(_QWORD *)(a2 - 48));
      v11 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
      if ( v11 )
      {
        if ( *(_QWORD *)(a2 - 48) )
        {
          v12 = *(_QWORD *)(a2 - 40);
          v13 = *(_QWORD *)(a2 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v13 = v12;
          if ( v12 )
            *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
        }
        *(_QWORD *)(a2 - 48) = v11;
        v14 = *(_QWORD *)(v11 + 8);
        *(_QWORD *)(a2 - 40) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = (a2 - 40) | *(_QWORD *)(v14 + 16) & 3LL;
        *(_QWORD *)(a2 - 32) = (v11 + 8) | *(_QWORD *)(a2 - 32) & 3LL;
        *(_QWORD *)(v11 + 8) = a2 - 48;
      }
      else if ( *(_QWORD *)(a2 - 48) )
      {
        v36 = *(_QWORD *)(a2 - 40);
        v37 = *(_QWORD *)(a2 - 32) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v37 = v36;
        if ( v36 )
          *(_QWORD *)(v36 + 16) = *(_QWORD *)(v36 + 16) & 3LL | v37;
        *(_QWORD *)(a2 - 48) = 0;
      }
      sub_16A85B0((__int64)&v44, (__int64)a3);
      v15 = sub_15A1070((__int64)v10, (__int64)&v44);
      if ( *(_QWORD *)(a2 - 24) )
      {
        v16 = *(_QWORD *)(a2 - 16);
        v17 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v17 = v16;
        if ( v16 )
          *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
      }
      *(_QWORD *)(a2 - 24) = v15;
      if ( v15 )
      {
        v18 = *(_QWORD *)(v15 + 8);
        *(_QWORD *)(a2 - 16) = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = (a2 - 16) | *(_QWORD *)(v18 + 16) & 3LL;
        *(_QWORD *)(a2 - 8) = (v15 + 8) | *(_QWORD *)(a2 - 8) & 3LL;
        *(_QWORD *)(v15 + 8) = a2 - 24;
      }
      if ( v45 > 0x40 && v44 )
        j_j___libc_free_0_0(v44);
      return a2;
    }
    if ( v9 != 31 )
      return v3;
  }
  v19 = (_QWORD **)*((unsigned int *)a3 + 2);
  v20 = (int)v19;
  if ( (unsigned int)v19 > 0x40 )
  {
    v40 = (_QWORD **)*((unsigned int *)a3 + 2);
    v35 = sub_16A57B0((__int64)a3);
    v19 = v40;
    if ( (unsigned int)(v20 - v35) > 0x40 )
      return 0;
    v21 = (_QWORD **)**a3;
  }
  else
  {
    v21 = (_QWORD **)*a3;
  }
  v3 = 0;
  if ( v19 == v21 )
  {
    sub_170B990(*a1, v4);
    v22 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    sub_1593B40((_QWORD *)(a2 - 48), v22);
    v25 = v10;
    goto LABEL_32;
  }
  return v3;
}

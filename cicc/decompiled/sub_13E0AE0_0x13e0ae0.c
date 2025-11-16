// Function: sub_13E0AE0
// Address: 0x13e0ae0
//
unsigned __int8 *__fastcall sub_13E0AE0(__int64 a1, unsigned __int8 *a2, __int64 a3, _QWORD *a4, int a5)
{
  unsigned int v7; // r13d
  unsigned __int8 *result; // rax
  bool v10; // al
  int v11; // eax
  unsigned __int8 v12; // al
  unsigned int v13; // r12d
  unsigned __int64 v14; // rcx
  unsigned int v15; // r13d
  unsigned int v16; // edx
  bool v18; // cc
  unsigned int v19; // edx
  __int64 v20; // rax
  char v21; // cl
  unsigned int v22; // edx
  int v23; // eax
  int v24; // eax
  __int64 **v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rax
  int v30; // [rsp+0h] [rbp-60h]
  int v31; // [rsp+4h] [rbp-5Ch]
  int v32; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v33; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v34; // [rsp+8h] [rbp-58h]
  unsigned int v35; // [rsp+8h] [rbp-58h]
  int v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v38; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v39; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v40; // [rsp+28h] [rbp-38h]

  v7 = a1;
  if ( a2[16] > 0x10u )
    goto LABEL_11;
  if ( *(_BYTE *)(a3 + 16) <= 0x10u )
  {
    result = (unsigned __int8 *)sub_14D6F90(a1, a2, a3, *a4);
    if ( result )
      return result;
    if ( a2[16] > 0x10u )
      goto LABEL_11;
  }
  if ( (unsigned __int8)sub_1593BB0(a2) )
    return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)a2);
  if ( a2[16] == 13 )
  {
    if ( *((_DWORD *)a2 + 8) <= 0x40u )
    {
      v10 = *((_QWORD *)a2 + 3) == 0;
    }
    else
    {
      v32 = *((_DWORD *)a2 + 8);
      v10 = v32 == (unsigned int)sub_16A57B0(a2 + 24);
    }
    goto LABEL_10;
  }
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
    goto LABEL_11;
  v28 = sub_15A1020(a2);
  if ( !v28 || *(_BYTE *)(v28 + 16) != 13 )
  {
    v31 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
    if ( v31 )
    {
      v19 = 0;
      while ( 1 )
      {
        v35 = v19;
        v20 = sub_15A0A60(a2, v19);
        if ( !v20 )
          goto LABEL_11;
        v21 = *(_BYTE *)(v20 + 16);
        v22 = v35;
        if ( v21 != 9 )
        {
          if ( v21 != 13 )
            goto LABEL_11;
          if ( *(_DWORD *)(v20 + 32) <= 0x40u )
          {
            if ( *(_QWORD *)(v20 + 24) )
              goto LABEL_11;
          }
          else
          {
            v30 = *(_DWORD *)(v20 + 32);
            v23 = sub_16A57B0(v20 + 24);
            v22 = v35;
            if ( v30 != v23 )
              goto LABEL_11;
          }
        }
        v19 = v22 + 1;
        if ( v31 == v19 )
          return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)a2);
      }
    }
    return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)a2);
  }
  if ( *(_DWORD *)(v28 + 32) <= 0x40u )
  {
    v10 = *(_QWORD *)(v28 + 24) == 0;
  }
  else
  {
    v36 = *(_DWORD *)(v28 + 32);
    v10 = v36 == (unsigned int)sub_16A57B0(v28 + 24);
  }
LABEL_10:
  if ( v10 )
    return (unsigned __int8 *)sub_15A06D0(*(_QWORD *)a2);
LABEL_11:
  if ( sub_13CD190(a3) )
    return a2;
  v11 = *(unsigned __int8 *)(a3 + 16);
  if ( (unsigned __int8)v11 > 0x17u )
  {
    v24 = v11 - 24;
  }
  else
  {
    if ( (_BYTE)v11 != 5 )
      goto LABEL_14;
    v24 = *(unsigned __int16 *)(a3 + 18);
  }
  if ( v24 == 38 )
  {
    v25 = (*(_BYTE *)(a3 + 23) & 0x40) != 0
        ? *(__int64 ***)(a3 - 8)
        : (__int64 **)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
    v26 = *v25;
    if ( v26 )
    {
      v27 = *v26;
      if ( *(_BYTE *)(*v26 + 8) == 16 )
        v27 = **(_QWORD **)(v27 + 16);
      if ( (unsigned __int8)sub_1642F90(v27, 1) )
        return a2;
    }
  }
LABEL_14:
  if ( sub_13CC0C0(a3) )
    return (unsigned __int8 *)sub_1599EF0(*(_QWORD *)a2);
  v12 = a2[16];
  if ( v12 == 79 || *(_BYTE *)(a3 + 16) == 79 )
  {
    result = sub_13DF4D0(v7, a2, (unsigned __int8 *)a3, a4, a5);
    if ( result )
      return result;
    v12 = a2[16];
  }
  if ( v12 != 77 && *(_BYTE *)(a3 + 16) != 77
    || (result = (unsigned __int8 *)sub_13DF6F0(v7, a2, (unsigned __int8 *)a3, a4, a5)) == 0 )
  {
    sub_14C2530((unsigned int)&v37, a3, *a4, 0, a4[3], a4[4], a4[2], 0);
    v13 = v40;
    if ( v40 > 0x40 )
    {
      if ( v13 - (unsigned int)sub_16A57B0(&v39) > 0x40 )
        goto LABEL_59;
      v14 = *v39;
    }
    else
    {
      v14 = (unsigned __int64)v39;
    }
    if ( v38 > v14 )
    {
      v15 = v38 - 1;
      if ( v38 != 1 )
      {
        _BitScanReverse(&v16, v15);
        v15 = 32 - (v16 ^ 0x1F);
        if ( v38 > 0x40 )
        {
          LODWORD(_RAX) = sub_16A58F0(&v37);
          goto LABEL_25;
        }
      }
      result = a2;
      _RDX = ~v37;
      if ( v37 != -1 )
      {
        __asm { tzcnt   rax, rdx }
LABEL_25:
        v18 = v15 <= (unsigned int)_RAX;
        result = 0;
        if ( v18 )
          result = a2;
      }
LABEL_27:
      if ( v13 > 0x40 && v39 )
      {
        v33 = result;
        j_j___libc_free_0_0(v39);
        result = v33;
      }
      if ( v38 > 0x40 )
      {
        if ( v37 )
        {
          v34 = result;
          j_j___libc_free_0_0(v37);
          return v34;
        }
      }
      return result;
    }
LABEL_59:
    result = (unsigned __int8 *)sub_1599EF0(*(_QWORD *)a2);
    v13 = v40;
    goto LABEL_27;
  }
  return result;
}

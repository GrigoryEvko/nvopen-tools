// Function: sub_16A8FA0
// Address: 0x16a8fa0
//
unsigned __int64 __fastcall sub_16A8FA0(__int64 a1, __int64 a2, unsigned int a3, char a4, char a5)
{
  char v6; // r14
  char *v7; // r12
  unsigned int v9; // edx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // r13
  unsigned __int64 result; // rax
  __int64 v13; // r15
  _BYTE *v14; // r12
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned int v17; // r14d
  unsigned int v18; // r15d
  int v19; // eax
  unsigned __int64 *v20; // rax
  char *v21; // r15
  __int64 v22; // rax
  char *v23; // r15
  unsigned int v24; // r14d
  unsigned __int8 *v25; // rdx
  unsigned __int8 *v26; // r12
  unsigned __int8 *v27; // rdx
  unsigned __int8 v28; // cl
  unsigned int v29; // esi
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  __int64 v33; // [rsp+0h] [rbp-A0h]
  unsigned int v34; // [rsp+8h] [rbp-98h]
  unsigned int v35; // [rsp+8h] [rbp-98h]
  unsigned int v37; // [rsp+Ch] [rbp-94h]
  unsigned __int64 v38; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v39; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v40; // [rsp+28h] [rbp-78h]
  _BYTE v41[63]; // [rsp+61h] [rbp-3Fh] BYREF

  v6 = 0;
  v7 = (char *)byte_3F871B3;
  if ( a5 )
  {
    if ( a3 != 10 )
    {
      v6 = 48;
      v7 = "0x";
      if ( a3 <= 0xA )
      {
        v7 = "0";
        if ( a3 == 2 )
          v7 = "0b";
      }
    }
  }
  v9 = *(_DWORD *)(a1 + 8);
  if ( v9 <= 0x40 )
  {
    v10 = *(unsigned int *)(a2 + 8);
    v11 = *(_QWORD *)a1;
    result = v10;
    if ( *(_QWORD *)a1 )
    {
      if ( a4 )
      {
        v13 = (__int64)(v11 << (64 - (unsigned __int8)v9)) >> (64 - (unsigned __int8)v9);
        v11 = v13;
        if ( v13 >= 0 )
        {
          if ( !v6 )
            goto LABEL_14;
          goto LABEL_11;
        }
        if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v10 )
        {
          sub_16CD150(a2, a2 + 16, 0, 1);
          v10 = *(unsigned int *)(a2 + 8);
        }
        v11 = -v13;
        *(_BYTE *)(*(_QWORD *)a2 + v10) = 45;
        v10 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v10;
      }
      if ( !v6 )
      {
LABEL_15:
        v14 = v41;
        do
        {
          *--v14 = a0123456789abcd_7[v11 % a3];
          v15 = v11;
          v11 /= a3;
        }
        while ( v15 >= a3 );
        v10 = (unsigned int)v10;
        v11 = v41 - v14;
        result = *(unsigned int *)(a2 + 12) - (unsigned __int64)(unsigned int)v10;
        if ( v41 - v14 > result )
        {
          result = sub_16CD150(a2, a2 + 16, v11 + (unsigned int)v10, 1);
          v10 = *(unsigned int *)(a2 + 8);
          if ( v14 == v41 )
            goto LABEL_20;
          goto LABEL_19;
        }
        if ( v14 != v41 )
        {
LABEL_19:
          result = (unsigned __int64)memcpy((void *)(*(_QWORD *)a2 + v10), v14, v41 - v14);
          LODWORD(v10) = *(_DWORD *)(a2 + 8);
        }
LABEL_20:
        *(_DWORD *)(a2 + 8) = v11 + v10;
        return result;
      }
      do
      {
LABEL_11:
        if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v10 )
        {
          sub_16CD150(a2, a2 + 16, 0, 1);
          v10 = *(unsigned int *)(a2 + 8);
        }
        ++v7;
        *(_BYTE *)(*(_QWORD *)a2 + v10) = v6;
        result = *(unsigned int *)(a2 + 8);
        v6 = *v7;
        v10 = (unsigned int)(result + 1);
        *(_DWORD *)(a2 + 8) = v10;
      }
      while ( v6 );
LABEL_14:
      if ( !v11 )
        goto LABEL_20;
      goto LABEL_15;
    }
LABEL_24:
    while ( v6 )
    {
      if ( *(_DWORD *)(a2 + 12) <= (unsigned int)result )
      {
        sub_16CD150(a2, a2 + 16, 0, 1);
        result = *(unsigned int *)(a2 + 8);
      }
      ++v7;
      *(_BYTE *)(*(_QWORD *)a2 + result) = v6;
      v6 = *v7;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
    }
    if ( *(_DWORD *)(a2 + 12) <= (unsigned int)result )
    {
      sub_16CD150(a2, a2 + 16, 0, 1);
      result = *(unsigned int *)(a2 + 8);
    }
    *(_BYTE *)(*(_QWORD *)a2 + result) = 48;
    ++*(_DWORD *)(a2 + 8);
    return result;
  }
  v34 = *(_DWORD *)(a1 + 8);
  if ( v9 - (unsigned int)sub_16A57B0(a1) <= 0x40 && !**(_QWORD **)a1 )
  {
    result = *(unsigned int *)(a2 + 8);
    goto LABEL_24;
  }
  v40 = v34;
  sub_16A4FD0((__int64)&v39, (const void **)a1);
  if ( !a4 )
    goto LABEL_33;
  v29 = *(_DWORD *)(a1 + 8);
  v30 = *(_QWORD *)a1;
  if ( v29 > 0x40 )
    v30 = *(_QWORD *)(v30 + 8LL * ((v29 - 1) >> 6));
  if ( (v30 & (1LL << ((unsigned __int8)v29 - 1))) != 0 )
  {
    if ( v40 <= 0x40 )
      v39 = ~v39 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v40);
    else
      sub_16A8F40((__int64 *)&v39);
    sub_16A7400((__int64)&v39);
    v31 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v31 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, a2 + 16, 0, 1);
      v31 = *(unsigned int *)(a2 + 8);
    }
    *(_BYTE *)(*(_QWORD *)a2 + v31) = 45;
    v16 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v16;
  }
  else
  {
LABEL_33:
    v16 = *(unsigned int *)(a2 + 8);
  }
  for ( ; v6; *(_DWORD *)(a2 + 8) = v16 )
  {
    if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v16 )
    {
      sub_16CD150(a2, a2 + 16, 0, 1);
      v16 = *(unsigned int *)(a2 + 8);
    }
    ++v7;
    *(_BYTE *)(*(_QWORD *)a2 + v16) = v6;
    v6 = *v7;
    v16 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  }
  v33 = (unsigned int)v16;
  if ( ((a3 - 8) & 0xFFFFFFF7) == 0 || a3 == 2 )
  {
    v17 = 4;
    if ( a3 != 16 )
      v17 = 2 * (a3 == 8) + 1;
    v18 = v40;
    v35 = a3 - 1;
LABEL_43:
    if ( v18 <= 0x40 )
    {
      while ( v39 )
      {
        v20 = &v39;
LABEL_46:
        v21 = &a0123456789abcd_7[*(_DWORD *)v20 & v35];
        if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v16 )
        {
          sub_16CD150(a2, a2 + 16, 0, 1);
          v16 = *(unsigned int *)(a2 + 8);
        }
        *(_BYTE *)(*(_QWORD *)a2 + v16) = *v21;
        v18 = v40;
        v16 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v16;
        if ( v18 > 0x40 )
        {
          sub_16A8110((__int64)&v39, v17);
          v16 = *(unsigned int *)(a2 + 8);
          v18 = v40;
          goto LABEL_43;
        }
        if ( v17 == v18 )
        {
          v39 = 0;
          goto LABEL_43;
        }
        v39 >>= v17;
      }
      result = *(_QWORD *)a2;
      v25 = (unsigned __int8 *)(*(_QWORD *)a2 + v16);
      v26 = (unsigned __int8 *)(*(_QWORD *)a2 + v33);
      if ( v26 == v25 )
        return result;
    }
    else
    {
      v37 = v16;
      v19 = sub_16A57B0((__int64)&v39);
      v16 = v37;
      if ( v19 != v18 )
      {
        v20 = (unsigned __int64 *)v39;
        goto LABEL_46;
      }
      result = *(_QWORD *)a2;
      v25 = (unsigned __int8 *)(*(_QWORD *)a2 + v37);
      v26 = (unsigned __int8 *)(*(_QWORD *)a2 + v33);
      if ( v26 == v25 )
        goto LABEL_69;
    }
LABEL_65:
    v27 = v25 - 1;
    v24 = v18;
    if ( v26 < v27 )
    {
      do
      {
        result = *v26;
        v28 = *v27;
        ++v26;
        --v27;
        *(v26 - 1) = v28;
        v27[1] = result;
      }
      while ( v26 < v27 );
      v24 = v40;
    }
    goto LABEL_68;
  }
  while ( 1 )
  {
    v18 = v40;
    v24 = v40;
    if ( v40 <= 0x40 )
      break;
    if ( v18 == (unsigned int)sub_16A57B0((__int64)&v39) )
      goto LABEL_64;
LABEL_59:
    sub_16A6DA0((__int64 **)&v39, a3, &v39, &v38);
    v22 = *(unsigned int *)(a2 + 8);
    v23 = &a0123456789abcd_7[v38];
    if ( (unsigned int)v22 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, a2 + 16, 0, 1);
      v22 = *(unsigned int *)(a2 + 8);
    }
    *(_BYTE *)(*(_QWORD *)a2 + v22) = *v23;
    ++*(_DWORD *)(a2 + 8);
  }
  if ( v39 )
    goto LABEL_59;
LABEL_64:
  result = *(_QWORD *)a2;
  v25 = (unsigned __int8 *)(*(_QWORD *)a2 + *(unsigned int *)(a2 + 8));
  v26 = (unsigned __int8 *)(*(_QWORD *)a2 + v33);
  if ( v26 != v25 )
    goto LABEL_65;
LABEL_68:
  if ( v24 > 0x40 )
  {
LABEL_69:
    if ( v39 )
      return j_j___libc_free_0_0(v39);
  }
  return result;
}

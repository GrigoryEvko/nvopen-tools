// Function: sub_D1D070
// Address: 0xd1d070
//
__int64 __fastcall sub_D1D070(__int64 a1, __int64 a2, unsigned __int8 **a3, __int64 a4)
{
  unsigned __int8 *v5; // rax
  unsigned __int8 *v6; // rbx
  __int64 v7; // r15
  unsigned __int8 **v8; // rax
  unsigned __int8 **v9; // rdx
  __int64 *v11; // r15
  int v12; // eax
  __int64 v13; // rcx
  unsigned __int64 v14; // rdx
  char v15; // di
  unsigned __int64 v16; // r8
  int v17; // r10d
  unsigned int v18; // r9d
  unsigned __int64 v19; // rsi
  unsigned __int8 *v20; // r11
  __int64 v21; // r9
  __int64 v22; // rsi
  int v23; // esi
  int v24; // r12d

  v5 = sub_98ACB0(*a3, 6u);
  if ( *v5 > 3u )
    return 3;
  v6 = v5;
  if ( (v5[32] & 0xFu) - 7 > 1 )
    return 3;
  if ( *(_BYTE *)(a1 + 136) )
    return 3;
  v7 = *(_QWORD *)(a2 - 32);
  if ( !v7 || *(_BYTE *)v7 || *(_QWORD *)(v7 + 24) != *(_QWORD *)(a2 + 80) )
    return 3;
  if ( *(_BYTE *)(a1 + 68) )
  {
    v8 = *(unsigned __int8 ***)(a1 + 48);
    v9 = &v8[*(unsigned int *)(a1 + 60)];
    if ( v8 == v9 )
      return 3;
    while ( v6 != *v8 )
    {
      if ( v9 == ++v8 )
        return 3;
    }
  }
  else if ( !sub_C8CA60(a1 + 40, (__int64)v5) )
  {
    return 3;
  }
  v11 = sub_D1B8E0(a1, v7);
  if ( !v11 )
    return 3;
  v12 = sub_D1C730(a1, (unsigned __int8 *)a2, v6, a4);
  v13 = ((unsigned __int64)*v11 >> 2) & 1;
  v14 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v14 )
    return (unsigned int)v13 | v12;
  v15 = *(_BYTE *)(v14 + 8) & 1;
  if ( v15 )
  {
    v16 = v14 + 16;
    v17 = 15;
  }
  else
  {
    v22 = *(unsigned int *)(v14 + 24);
    v16 = *(_QWORD *)(v14 + 16);
    if ( !(_DWORD)v22 )
      goto LABEL_31;
    v17 = v22 - 1;
  }
  v18 = v17 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v19 = v16 + 16LL * v18;
  v20 = *(unsigned __int8 **)v19;
  if ( v6 == *(unsigned __int8 **)v19 )
    goto LABEL_21;
  v23 = 1;
  while ( v20 != (unsigned __int8 *)-4096LL )
  {
    v24 = v23 + 1;
    v18 = v17 & (v23 + v18);
    v19 = v16 + 16LL * v18;
    v20 = *(unsigned __int8 **)v19;
    if ( v6 == *(unsigned __int8 **)v19 )
      goto LABEL_21;
    v23 = v24;
  }
  v22 = 16;
  if ( !v15 )
    v22 = *(unsigned int *)(v14 + 24);
LABEL_31:
  v19 = v16 + 16 * v22;
LABEL_21:
  v21 = 256;
  if ( !v15 )
    v21 = 16LL * *(unsigned int *)(v14 + 24);
  if ( v19 == v21 + v16 )
    return (unsigned int)v13 | v12;
  LOBYTE(v13) = *(_BYTE *)(v19 + 8) | ((*v11 & 4) != 0);
  return (unsigned int)v13 | v12;
}

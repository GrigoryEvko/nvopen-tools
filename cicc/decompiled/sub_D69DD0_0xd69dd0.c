// Function: sub_D69DD0
// Address: 0xd69dd0
//
__int64 __fastcall sub_D69DD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rbx
  _QWORD *v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rax
  int v13; // r15d
  __int64 v14; // rdx
  int v15; // ecx
  char v16; // dl
  __int64 v17; // rsi
  int v18; // ecx
  unsigned int v19; // eax
  __int64 *v20; // r15
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 *v23; // rbx
  __int64 *v24; // r12
  __int64 v25; // rsi
  __int64 v27; // rax
  __int64 v28; // r15
  int v29; // r8d

  v9 = *(_QWORD *)(a3 + 16);
  if ( v9 )
  {
    while ( (unsigned __int8)(**(_BYTE **)(v9 + 24) - 30) > 0xAu )
    {
      v9 = *(_QWORD *)(v9 + 8);
      if ( !v9 )
        goto LABEL_27;
    }
    v10 = (_QWORD *)(a1 + 16);
    v11 = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0x800000000LL;
    v12 = v9;
    while ( 1 )
    {
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
        break;
      while ( (unsigned __int8)(**(_BYTE **)(v12 + 24) - 30) <= 0xAu )
      {
        v12 = *(_QWORD *)(v12 + 8);
        ++v11;
        if ( !v12 )
          goto LABEL_7;
      }
    }
LABEL_7:
    v13 = v11 + 1;
    if ( v11 + 1 > 8 )
    {
      sub_C8D5F0(a1, v10, v11 + 1, 8u, a5, a6);
      v10 = (_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
    }
    v14 = *(_QWORD *)(v9 + 24);
LABEL_12:
    if ( v10 )
      *v10 = *(_QWORD *)(v14 + 40);
    while ( 1 )
    {
      v9 = *(_QWORD *)(v9 + 8);
      if ( !v9 )
        break;
      v14 = *(_QWORD *)(v9 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v14 - 30) <= 0xAu )
      {
        ++v10;
        goto LABEL_12;
      }
    }
    v15 = v13 + *(_DWORD *)(a1 + 8);
  }
  else
  {
LABEL_27:
    *(_DWORD *)(a1 + 12) = 8;
    v15 = 0;
    *(_QWORD *)a1 = a1 + 16;
  }
  *(_DWORD *)(a1 + 8) = v15;
  sub_B1C8F0(a1);
  v16 = *(_BYTE *)(a2 + 312) & 1;
  if ( v16 )
  {
    v17 = a2 + 320;
    v18 = 3;
  }
  else
  {
    v27 = *(unsigned int *)(a2 + 328);
    v17 = *(_QWORD *)(a2 + 320);
    if ( !(_DWORD)v27 )
      goto LABEL_31;
    v18 = v27 - 1;
  }
  v19 = v18 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v20 = (__int64 *)(v17 + 72LL * v19);
  v21 = *v20;
  if ( *v20 == a3 )
    goto LABEL_19;
  v29 = 1;
  while ( v21 != -4096 )
  {
    v19 = v18 & (v29 + v19);
    v20 = (__int64 *)(v17 + 72LL * v19);
    v21 = *v20;
    if ( *v20 == a3 )
      goto LABEL_19;
    ++v29;
  }
  if ( v16 )
  {
    v28 = 288;
    goto LABEL_32;
  }
  v27 = *(unsigned int *)(a2 + 328);
LABEL_31:
  v28 = 72 * v27;
LABEL_32:
  v20 = (__int64 *)(v17 + v28);
LABEL_19:
  v22 = 288;
  if ( !v16 )
    v22 = 72LL * *(unsigned int *)(a2 + 328);
  if ( v20 != (__int64 *)(v17 + v22) )
  {
    v23 = (__int64 *)v20[1];
    v24 = &v23[*((unsigned int *)v20 + 4)];
    while ( v24 != v23 )
    {
      v25 = *v23++;
      sub_B1CA60(a1, v25);
    }
    sub_D67E60(
      a1,
      (char *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8)),
      (char *)v20[5],
      (char *)(v20[5] + 8LL * *((unsigned int *)v20 + 12)));
  }
  return a1;
}

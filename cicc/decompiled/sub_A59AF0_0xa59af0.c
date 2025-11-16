// Function: sub_A59AF0
// Address: 0xa59af0
//
void __fastcall sub_A59AF0(__int64 a1, _BYTE *a2)
{
  __int64 v3; // r12
  unsigned int v5; // esi
  __int64 v6; // r8
  int v7; // r10d
  __int64 *v8; // rcx
  unsigned int v9; // edx
  __int64 *v10; // rax
  _BYTE *v11; // rdi
  __int64 v12; // rdi
  int v13; // edx
  unsigned __int8 v14; // al
  int v15; // ecx
  __int64 v16; // r14
  __int64 v17; // r15
  _BYTE *i; // rax
  _BYTE *v19; // rsi
  int v20; // eax
  __int64 *v21; // [rsp-50h] [rbp-50h] BYREF
  _BYTE *v22; // [rsp-48h] [rbp-48h] BYREF
  int v23; // [rsp-40h] [rbp-40h]

  if ( *a2 == 7 )
    return;
  v3 = a1 + 184;
  v22 = a2;
  v5 = *(_DWORD *)(a1 + 208);
  v23 = *(_DWORD *)(a1 + 216);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 184);
    v21 = 0;
LABEL_7:
    v5 *= 2;
LABEL_8:
    sub_A59910(v3, v5);
    sub_A57020(v3, (__int64 *)&v22, &v21);
    v12 = (__int64)v22;
    v8 = v21;
    v13 = *(_DWORD *)(a1 + 200) + 1;
    goto LABEL_9;
  }
  v6 = *(_QWORD *)(a1 + 192);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = (_BYTE *)*v10;
  if ( a2 == (_BYTE *)*v10 )
    return;
  while ( v11 != (_BYTE *)-4096LL )
  {
    if ( v8 || v11 != (_BYTE *)-8192LL )
      v10 = v8;
    v9 = (v5 - 1) & (v7 + v9);
    v11 = *(_BYTE **)(v6 + 16LL * v9);
    if ( a2 == v11 )
      return;
    ++v7;
    v8 = v10;
    v10 = (__int64 *)(v6 + 16LL * v9);
  }
  if ( !v8 )
    v8 = v10;
  v20 = *(_DWORD *)(a1 + 200);
  ++*(_QWORD *)(a1 + 184);
  v13 = v20 + 1;
  v21 = v8;
  if ( 4 * (v20 + 1) >= 3 * v5 )
    goto LABEL_7;
  v12 = (__int64)a2;
  if ( v5 - *(_DWORD *)(a1 + 204) - v13 <= v5 >> 3 )
    goto LABEL_8;
LABEL_9:
  *(_DWORD *)(a1 + 200) = v13;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 204);
  *v8 = v12;
  *((_DWORD *)v8 + 2) = v23;
  ++*(_DWORD *)(a1 + 216);
  v14 = *(a2 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *((_DWORD *)a2 - 6);
  else
    v15 = (*((_WORD *)a2 - 8) >> 6) & 0xF;
  if ( v15 )
  {
    v16 = 0;
    v17 = 8LL * (unsigned int)(v15 - 1);
    if ( (*(a2 - 16) & 2) == 0 )
      goto LABEL_21;
LABEL_15:
    for ( i = (_BYTE *)*((_QWORD *)a2 - 4); ; i = &a2[-16 - 8LL * ((v14 >> 2) & 0xF)] )
    {
      v19 = *(_BYTE **)&i[v16];
      if ( v19 )
      {
        if ( (unsigned __int8)(*v19 - 5) <= 0x1Fu )
          sub_A59AF0(a1);
      }
      if ( v16 == v17 )
        break;
      v14 = *(a2 - 16);
      v16 += 8;
      if ( (v14 & 2) != 0 )
        goto LABEL_15;
LABEL_21:
      ;
    }
  }
}

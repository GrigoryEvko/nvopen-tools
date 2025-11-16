// Function: sub_F2FEA0
// Address: 0xf2fea0
//
__int64 __fastcall sub_F2FEA0(_BYTE *a1, _BYTE *a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r9d
  unsigned int v10; // edx
  __int64 *v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 *v14; // rsi
  __int64 v15; // rdi
  int v17; // esi
  int v18; // r10d
  int v19; // esi
  int v20; // r10d

  v6 = sub_B326A0((__int64)a2);
  if ( !v6 )
  {
    if ( (a2[32] & 0xFu) - 7 <= 1 || sub_F2FDA0((__int64)a1, a2) )
      return 0;
    goto LABEL_15;
  }
  v7 = *(unsigned int *)(a3 + 24);
  v8 = *(_QWORD *)(a3 + 8);
  if ( (_DWORD)v7 )
  {
    v9 = v7 - 1;
    v10 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v11 = (__int64 *)(v8 + 24LL * v10);
    v12 = *v11;
    if ( v6 == *v11 )
    {
LABEL_4:
      if ( *((_BYTE *)v11 + 16) )
        return 0;
    }
    else
    {
      v17 = 1;
      while ( v12 != -4096 )
      {
        v18 = v17 + 1;
        v10 = v9 & (v17 + v10);
        v11 = (__int64 *)(v8 + 24LL * v10);
        v12 = *v11;
        if ( v6 == *v11 )
          goto LABEL_4;
        v17 = v18;
      }
    }
    if ( (unsigned __int8)(*a2 - 2) > 1u && *a2 )
      goto LABEL_11;
    v13 = v9 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v14 = (__int64 *)(v8 + 24 * v13);
    v15 = *v14;
    if ( v6 == *v14 )
      goto LABEL_8;
    v19 = 1;
    while ( v15 != -4096 )
    {
      v20 = v19 + 1;
      v13 = v9 & (unsigned int)(v19 + v13);
      v14 = (__int64 *)(v8 + 24LL * (unsigned int)v13);
      v15 = *v14;
      if ( v6 == *v14 )
        goto LABEL_8;
      v19 = v20;
    }
LABEL_18:
    v13 = 3 * v7;
    v14 = (__int64 *)(v8 + 24 * v7);
LABEL_8:
    if ( v14[1] == 1 )
    {
      sub_B2F990((__int64)a2, 0, v13, v7);
    }
    else if ( !*a1 )
    {
      *(_DWORD *)(v6 + 8) = 3;
    }
    goto LABEL_11;
  }
  if ( (unsigned __int8)(*a2 - 2) <= 1u || !*a2 )
    goto LABEL_18;
LABEL_11:
  if ( (a2[32] & 0xFu) - 7 <= 1 )
    return 0;
LABEL_15:
  *((_WORD *)a2 + 16) = *((_WORD *)a2 + 16) & 0xBCC0 | 0x4007;
  return 1;
}

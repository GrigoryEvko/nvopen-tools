// Function: sub_8C7610
// Address: 0x8c7610
//
_BOOL8 __fastcall sub_8C7610(__int64 a1)
{
  __int64 v1; // r12
  __int64 *v3; // rax
  _BOOL8 result; // rax
  __int64 *v5; // rdx
  __int64 *v6; // rsi
  __int64 *v7; // rdi
  __int64 **v8; // rdx
  __int64 *v9; // rsi
  __int64 *v10; // rdi
  __int64 **v11; // rdx
  __int64 *v12; // rdx
  __int64 **v13; // rdx
  __int64 **v14; // rcx

  v1 = a1;
  v3 = *(__int64 **)(a1 + 32);
  if ( v3 )
    v1 = *v3;
  result = sub_8C7520((__int64 **)a1, (__int64 **)v1);
  if ( !result )
  {
    v5 = *(__int64 **)a1;
    v6 = *(__int64 **)v1;
    if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    {
      if ( v5
        && v6
        && *((_BYTE *)v5 + 80) == 10
        && *((_BYTE *)v6 + 80) == 10
        && *(_BYTE *)(v5[11] + 174) == 3
        && *(_BYTE *)(v6[11] + 174) == 3 )
      {
        return 1;
      }
      v7 = *(__int64 **)(*(_QWORD *)(a1 + 40) + 32LL);
      v8 = (__int64 **)v7[4];
      if ( v8 )
      {
        v9 = *v8;
        if ( v7 != *v8 || (*(_BYTE *)(v1 + 89) & 4) == 0 )
          goto LABEL_10;
      }
      else
      {
        v9 = *(__int64 **)(*(_QWORD *)(a1 + 40) + 32LL);
        if ( (*(_BYTE *)(v1 + 89) & 4) == 0 )
        {
LABEL_10:
          sub_8C6700(v7, (unsigned int *)v9 + 16, 0x42Au, 0x425u);
          return 0;
        }
      }
      v9 = *(__int64 **)(*(_QWORD *)(v1 + 40) + 32LL);
      v13 = (__int64 **)v9[4];
      v7 = v9;
      if ( v13 )
        v9 = *v13;
      goto LABEL_10;
    }
    if ( (v5[10] & 0x10FF) == 0x100A && (*(_BYTE *)(v5[11] + 195) & 8) != 0 )
    {
      v10 = (__int64 *)v5[8];
      v11 = (__int64 **)v10[4];
      if ( v11 )
      {
        v12 = *v11;
        if ( v10 != v12 || (*((_BYTE *)v6 + 81) & 0x10) == 0 )
          goto LABEL_16;
      }
      else
      {
        v12 = v10;
        if ( (*((_BYTE *)v6 + 81) & 0x10) == 0 )
        {
LABEL_16:
          sub_8C6700(v10, (unsigned int *)v12 + 16, 0x42Au, 0x425u);
          return 0;
        }
      }
      v12 = (__int64 *)v6[8];
      v14 = (__int64 **)v12[4];
      v10 = v12;
      if ( v14 )
        v12 = *v14;
      goto LABEL_16;
    }
  }
  return result;
}

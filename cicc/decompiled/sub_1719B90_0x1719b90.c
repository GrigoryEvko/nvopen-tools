// Function: sub_1719B90
// Address: 0x1719b90
//
__int64 __fastcall sub_1719B90(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  char v6; // al
  __int16 v7; // ax
  __int64 v9; // rax
  _BYTE *v10; // rdi
  unsigned __int8 v11; // al
  __int64 v12; // r13
  unsigned int v13; // ecx
  __int64 v14; // rax
  _BYTE *v15; // rdi
  unsigned __int8 v16; // al
  __int64 v17; // rsi
  __int64 v18; // rdx
  char v19; // al
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v30; // [rsp+18h] [rbp-28h]

  v6 = *(_BYTE *)(a1 + 16);
  switch ( v6 )
  {
    case 39:
      v14 = *(_QWORD *)(a1 - 48);
      if ( !v14 )
        return 0;
      *a2 = v14;
      v15 = *(_BYTE **)(a1 - 24);
      v16 = v15[16];
      if ( v16 != 13 )
      {
        v18 = *(_QWORD *)v15;
        if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 16 && v16 <= 0x10u )
        {
LABEL_41:
          v27 = sub_15A1020(v15, (__int64)a2, v18, a4);
          if ( v27 && *(_BYTE *)(v27 + 16) == 13 )
          {
            v17 = v27 + 24;
            goto LABEL_19;
          }
        }
LABEL_24:
        v19 = *(_BYTE *)(a1 + 16);
        if ( v19 != 47 )
        {
          if ( v19 != 5 )
            return 0;
          v7 = *(_WORD *)(a1 + 18);
LABEL_4:
          if ( v7 != 23 )
            return 0;
          v21 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
          if ( !v21 )
            return 0;
          *a2 = v21;
          v20 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
          v10 = *(_BYTE **)(a1 + 24 * (1 - v20));
          if ( v10[16] != 13 )
          {
            if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 )
              return 0;
            goto LABEL_33;
          }
          goto LABEL_9;
        }
LABEL_7:
        v9 = *(_QWORD *)(a1 - 48);
        if ( !v9 )
          return 0;
        *a2 = v9;
        v10 = *(_BYTE **)(a1 - 24);
        v11 = v10[16];
        if ( v11 != 13 )
        {
          v20 = *(_QWORD *)v10;
          if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) != 16 || v11 > 0x10u )
            return 0;
LABEL_33:
          v22 = sub_15A1020(v10, (__int64)a2, v20, a4);
          if ( v22 && *(_BYTE *)(v22 + 16) == 13 )
          {
            v12 = v22 + 24;
            goto LABEL_10;
          }
          return 0;
        }
LABEL_9:
        v12 = (__int64)(v10 + 24);
LABEL_10:
        v13 = *(_DWORD *)(v12 + 8);
        v30 = v13;
        if ( v13 > 0x40 )
          sub_16A4EF0((__int64)&v29, 1, 0);
        else
          v29 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v13) & 1;
        if ( *(_DWORD *)(a3 + 8) > 0x40u )
        {
          if ( *(_QWORD *)a3 )
            j_j___libc_free_0_0(*(_QWORD *)a3);
        }
        *(_QWORD *)a3 = v29;
        *(_DWORD *)(a3 + 8) = v30;
        sub_16A7E20(a3, v12);
        return 1;
      }
      break;
    case 5:
      v7 = *(_WORD *)(a1 + 18);
      if ( v7 != 15 )
        goto LABEL_4;
      v26 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      if ( !v26 )
        return 0;
      *a2 = v26;
      v18 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
      v15 = *(_BYTE **)(a1 + 24 * (1 - v18));
      if ( v15[16] != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 16 )
          goto LABEL_41;
        goto LABEL_24;
      }
      break;
    case 47:
      goto LABEL_7;
    default:
      return 0;
  }
  v17 = (__int64)(v15 + 24);
LABEL_19:
  if ( *(_DWORD *)(a3 + 8) <= 0x40u && *(_DWORD *)(v17 + 8) <= 0x40u )
  {
    v23 = *(_QWORD *)v17;
    *(_QWORD *)a3 = *(_QWORD *)v17;
    v24 = *(unsigned int *)(v17 + 8);
    *(_DWORD *)(a3 + 8) = v24;
    v25 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v24;
    if ( (unsigned int)v24 > 0x40 )
    {
      v28 = (unsigned int)((unsigned __int64)(v24 + 63) >> 6) - 1;
      *(_QWORD *)(v23 + 8 * v28) &= v25;
      return 1;
    }
    else
    {
      *(_QWORD *)a3 = v25 & v23;
      return 1;
    }
  }
  else
  {
    sub_16A51C0(a3, v17);
    return 1;
  }
}

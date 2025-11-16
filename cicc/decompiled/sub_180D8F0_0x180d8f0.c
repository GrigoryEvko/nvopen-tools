// Function: sub_180D8F0
// Address: 0x180d8f0
//
_BYTE *__fastcall sub_180D8F0(__int64 a1, __int64 a2, _BYTE *a3, unsigned __int64 *a4, int *a5, _QWORD *a6)
{
  __int64 v11; // rax
  __int64 v12; // r9
  char v13; // al
  __int64 v14; // rsi
  _BYTE *v15; // r15
  __int64 v16; // rax
  __int64 v18; // rdi
  const char *v19; // rax
  __int64 v20; // r9
  unsigned __int64 v21; // rdx
  const char *v22; // rax
  unsigned __int64 v23; // rdx
  const char *v24; // rax
  __int64 v25; // r9
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+8h] [rbp-48h]
  int v31; // [rsp+8h] [rbp-48h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+10h] [rbp-40h]

  if ( (*(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0) && sub_1625940(a2, "nosanitize", 0xAu)
    || *(_QWORD *)(a1 + 720) == a2 )
  {
    return 0;
  }
  v11 = sub_15F2050(a2);
  v12 = sub_1632FA0(v11);
  v13 = *(_BYTE *)(a2 + 16);
  switch ( v13 )
  {
    case '6':
      if ( !byte_4FA8C80 )
        return 0;
      *a3 = 0;
      v14 = *(_QWORD *)a2;
LABEL_7:
      *a4 = (sub_127FA20(v12, v14) + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *a5 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
      v15 = *(_BYTE **)(a2 - 24);
      if ( v15 )
        break;
      return 0;
    case '7':
      if ( !byte_4FA8BA0 )
        return 0;
      *a3 = 1;
      v14 = **(_QWORD **)(a2 - 48);
      goto LABEL_7;
    case ';':
      if ( !byte_4FA8AC0 )
        return 0;
      *a3 = 1;
      *a4 = (sub_127FA20(v12, **(_QWORD **)(a2 - 24)) + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *a5 = 0;
      v15 = *(_BYTE **)(a2 - 48);
      if ( !v15 )
        return 0;
      break;
    case ':':
      if ( !byte_4FA8AC0 )
        return 0;
      *a3 = 1;
      *a4 = (sub_127FA20(v12, **(_QWORD **)(a2 - 48)) + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *a5 = 0;
      v15 = *(_BYTE **)(a2 - 72);
      if ( !v15 )
        return 0;
      break;
    default:
      v32 = v12;
      if ( v13 != 78 )
        return 0;
      v18 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v18 + 16) )
        return 0;
      v30 = *(_QWORD *)(a2 - 24);
      v19 = sub_1649960(v18);
      v20 = v32;
      if ( v21 <= 0x10
        || *(_QWORD *)v19 ^ 0x73616D2E6D766C6CLL | *((_QWORD *)v19 + 1) ^ 0x64616F6C2E64656BLL
        || v19[16] != 46 )
      {
        v22 = sub_1649960(v30);
        if ( v23 <= 0x11 )
          return 0;
        if ( *(_QWORD *)v22 ^ 0x73616D2E6D766C6CLL | *((_QWORD *)v22 + 1) ^ 0x726F74732E64656BLL )
          return 0;
        v20 = v32;
        if ( *((_WORD *)v22 + 8) != 11877 )
          return 0;
      }
      v33 = v20;
      v24 = sub_1649960(v30);
      v25 = v33;
      if ( v26 > 0x11
        && !(*(_QWORD *)v24 ^ 0x73616D2E6D766C6CLL | *((_QWORD *)v24 + 1) ^ 0x726F74732E64656BLL)
        && *((_WORD *)v24 + 8) == 11877 )
      {
        if ( !byte_4FA8BA0 )
          return 0;
        *a3 = 1;
        v27 = 1;
        v34 = 2;
        v31 = 1;
      }
      else
      {
        if ( !byte_4FA8C80 )
          return 0;
        *a3 = 0;
        v27 = 0;
        v34 = 1;
        v31 = 0;
      }
      v15 = *(_BYTE **)(a2 + 24 * (v27 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      *a4 = (sub_127FA20(v25, *(_QWORD *)(*(_QWORD *)v15 + 24LL)) + 7) & 0xFFFFFFFFFFFFFFF8LL;
      v28 = *(_QWORD *)(a2 + 24 * (v34 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v28 + 16) == 13 )
      {
        if ( *(_DWORD *)(v28 + 32) <= 0x40u )
          v29 = *(_QWORD *)(v28 + 24);
        else
          v29 = **(_QWORD **)(v28 + 24);
        *a5 = v29;
      }
      else
      {
        *a5 = 1;
      }
      if ( a6 )
        *a6 = *(_QWORD *)(a2 + 24 * ((unsigned int)(v31 + 2) - (unsigned __int64)(*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      break;
  }
  v16 = *(_QWORD *)v15;
  if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 16 )
  {
    v16 = **(_QWORD **)(v16 + 16);
    if ( *(_BYTE *)(v16 + 8) == 16 )
      v16 = **(_QWORD **)(v16 + 16);
  }
  if ( *(_DWORD *)(v16 + 8) >> 8
    || (unsigned __int8)sub_1649A90((__int64)v15)
    || byte_4FA7AC0 && v15[16] == 53 && !(unsigned __int8)sub_180D640(a1, (__int64)v15) )
  {
    return 0;
  }
  return v15;
}

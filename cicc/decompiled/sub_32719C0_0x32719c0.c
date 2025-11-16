// Function: sub_32719C0
// Address: 0x32719c0
//
__int64 __fastcall sub_32719C0(_DWORD *a1, __int64 a2, unsigned __int64 a3, char a4)
{
  unsigned __int64 v4; // r15
  unsigned int v7; // ebx
  int v8; // eax
  char v9; // al
  __int64 *v11; // rax
  __int64 *v12; // rcx
  __int64 v13; // rcx
  unsigned __int16 *v14; // rsi
  __int64 v15; // rax
  unsigned __int16 v16; // ax
  __int64 v17; // rcx
  unsigned __int16 v18; // cx
  int v19; // eax
  char v20; // bl
  char v22; // [rsp+Fh] [rbp-41h]
  unsigned __int16 v23; // [rsp+10h] [rbp-40h] BYREF
  __int64 v24; // [rsp+18h] [rbp-38h]

  v4 = a3;
  v7 = a3;
  v22 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v8 = *(_DWORD *)(a2 + 24);
      if ( ((v8 - 214) & 0xFFFFFFFD) != 0 )
        break;
      v11 = *(__int64 **)(a2 + 40);
      a2 = *v11;
      v7 = *((_DWORD *)v11 + 2);
      v4 = v7 | v4 & 0xFFFFFFFF00000000LL;
    }
    if ( v8 != 186 )
      break;
    v9 = sub_33CF4D0(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
    if ( !v9 )
      break;
    if ( a4 )
      return a2;
    v12 = *(__int64 **)(a2 + 40);
    v22 = v9;
    a2 = *v12;
    v7 = *((_DWORD *)v12 + 2);
    v4 = v7 | v4 & 0xFFFFFFFF00000000LL;
  }
  if ( a4 && *(_WORD *)(*(_QWORD *)(a2 + 48) + 16LL * v7) == 2 )
    return a2;
  if ( v7 != 1 )
    return 0;
  v13 = *(unsigned int *)(a2 + 24);
  if ( (*(_DWORD *)(a2 + 24) & 0xFFFFFFFD) != 0x4D && (unsigned int)(v13 - 72) > 1 )
    return 0;
  v14 = *(unsigned __int16 **)(a2 + 48);
  v15 = *v14;
  if ( (_WORD)v15 != 1 && (!(_WORD)v15 || !*(_QWORD *)&a1[2 * (unsigned __int16)v15 + 28]) )
    return 0;
  if ( (unsigned int)v13 <= 0x1F3 && (*((_BYTE *)&a1[125 * v15 + 1603] + v13 + 2) & 0xFB) != 0 )
    return 0;
  if ( !v22 )
  {
    v16 = v14[8];
    v17 = *((_QWORD *)v14 + 3);
    v23 = v16;
    v24 = v17;
    if ( v16 )
    {
      v18 = v16 - 17;
      if ( (unsigned __int16)(v16 - 10) > 6u && (unsigned __int16)(v16 - 126) > 0x31u )
      {
        if ( v18 > 0xD3u )
        {
LABEL_26:
          v19 = a1[15];
          goto LABEL_27;
        }
        goto LABEL_33;
      }
      if ( v18 <= 0xD3u )
      {
LABEL_33:
        v19 = a1[17];
LABEL_27:
        if ( v19 == 1 )
          return a2;
        return 0;
      }
    }
    else
    {
      v20 = sub_3007030((__int64)&v23);
      if ( sub_30070B0((__int64)&v23) )
        goto LABEL_33;
      if ( !v20 )
        goto LABEL_26;
    }
    v19 = a1[16];
    goto LABEL_27;
  }
  return a2;
}

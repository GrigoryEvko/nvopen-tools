// Function: sub_1F6DE40
// Address: 0x1f6de40
//
__int64 __fastcall sub_1F6DE40(_DWORD *a1, __int64 a2, unsigned __int64 a3)
{
  bool v3; // r15
  unsigned int v4; // r13d
  unsigned __int64 i; // r12
  __int16 v7; // ax
  bool v8; // al
  __int64 *v10; // rax
  __int64 *v11; // rcx
  int v12; // eax
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // rdx
  bool v16; // r14
  int v17; // eax
  _BYTE v18[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v19; // [rsp+18h] [rbp-38h]

  v3 = 0;
  v4 = a3;
  for ( i = a3; ; i = v4 | i & 0xFFFFFFFF00000000LL )
  {
    while ( 1 )
    {
      v7 = *(_WORD *)(a2 + 24);
      if ( ((v7 - 143) & 0xFFFD) != 0 )
        break;
      v10 = *(__int64 **)(a2 + 32);
      a2 = *v10;
      v4 = *((_DWORD *)v10 + 2);
      i = v4 | i & 0xFFFFFFFF00000000LL;
    }
    if ( v7 != 118 )
      break;
    v8 = sub_1D18910(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
    if ( !v8 )
      break;
    v11 = *(__int64 **)(a2 + 32);
    v3 = v8;
    a2 = *v11;
    v4 = *((_DWORD *)v11 + 2);
  }
  if ( v4 != 1 )
    return 0;
  v12 = *(unsigned __int16 *)(a2 + 24);
  if ( (((_WORD)v12 - 71) & 0xFFFD) != 0 && (unsigned int)(v12 - 68) > 1 )
    return 0;
  if ( !v3 )
  {
    v13 = *(_QWORD *)(a2 + 40);
    v14 = *(_BYTE *)(v13 + 16);
    v15 = *(_QWORD *)(v13 + 24);
    v18[0] = v14;
    v19 = v15;
    if ( v14 )
    {
      if ( (unsigned __int8)(v14 - 14) > 0x5Fu )
      {
        v16 = (unsigned __int8)(v14 - 86) <= 0x17u || (unsigned __int8)(v14 - 8) <= 5u;
        goto LABEL_15;
      }
    }
    else
    {
      v16 = sub_1F58CD0((__int64)v18);
      if ( !sub_1F58D20((__int64)v18) )
      {
LABEL_15:
        if ( v16 )
          v17 = a1[16];
        else
          v17 = a1[15];
LABEL_17:
        if ( v17 == 1 )
          return a2;
        return 0;
      }
    }
    v17 = a1[17];
    goto LABEL_17;
  }
  return a2;
}

// Function: sub_8B2240
// Address: 0x8b2240
//
__int64 **__fastcall sub_8B2240(__int64 *a1, __int64 a2, _QWORD *a3, unsigned int a4, __int64 a5)
{
  _QWORD *v5; // r14
  __int64 v8; // r15
  __int64 v9; // rdi
  _QWORD *v10; // rax
  __int64 v11; // r9
  __int64 **v12; // r14
  __m128i *v14; // rax
  _QWORD *v15; // rdx
  unsigned int v16; // [rsp+8h] [rbp-38h]
  _QWORD *v17; // [rsp+8h] [rbp-38h]

  v5 = a3;
  switch ( *(_BYTE *)(a2 + 80) )
  {
    case 4:
    case 5:
      v8 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 80LL);
      goto LABEL_3;
    case 6:
      v8 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 32LL);
      goto LABEL_3;
    case 9:
    case 0xA:
      v8 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 56LL);
      if ( !a3 )
        goto LABEL_14;
      goto LABEL_4;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v8 = *(_QWORD *)(a2 + 88);
      goto LABEL_3;
    default:
      v8 = 0;
LABEL_3:
      if ( a3 )
      {
LABEL_4:
        v9 = *a1;
        if ( *a1 )
          goto LABEL_5;
      }
      else
      {
LABEL_14:
        v9 = *a1;
        v5 = **(_QWORD ***)(v8 + 328);
        if ( *a1 )
          goto LABEL_5;
      }
      v16 = a5;
      v14 = sub_8A3C00((__int64)v5, 0, 0, (__int64 *)(a2 + 48));
      a5 = v16;
      *a1 = (__int64)v14;
      v9 = (__int64)v14;
LABEL_5:
      if ( !(unsigned int)sub_8B59E0(v9, a2, v5, a4, a5) )
        return 0;
      if ( unk_4F04C48 != -1 )
      {
        v10 = (_QWORD *)(qword_4F04C68[0] + 776LL * unk_4F04C48);
        if ( v10[46] == a2 && !v10[47] )
          v10[48] = *a1;
      }
      v12 = sub_8B1C20(a2, *a1, 0, v5, a4);
      if ( !v12 )
        return 0;
      if ( !dword_4F077BC )
        goto LABEL_10;
      if ( (unsigned int)sub_8D5830(v12[20]) )
        return 0;
      v15 = (_QWORD *)*v12[21];
      if ( !v15 )
        goto LABEL_10;
      break;
  }
  do
  {
    v17 = v15;
    if ( (unsigned int)sub_8D5830(v15[1]) )
      return 0;
    v15 = (_QWORD *)*v17;
  }
  while ( *v17 );
LABEL_10:
  if ( (a4 & 8) == 0 )
    sub_894B30(a2, v8, (const __m128i *)*a1, a4, (__int64)v12, v11);
  return v12;
}

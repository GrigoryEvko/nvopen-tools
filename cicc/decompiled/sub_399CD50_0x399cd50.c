// Function: sub_399CD50
// Address: 0x399cd50
//
_QWORD *__fastcall sub_399CD50(_QWORD *a1, __int64 a2, __int64 *a3, unsigned int a4)
{
  __int64 *v5; // rdi
  __int64 v8; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int8 *v15; // r14
  unsigned __int16 v16; // ax
  unsigned __int8 *v17; // rsi
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = (__int64 *)a3[41];
  if ( v5 == a3 + 40 )
    goto LABEL_9;
  while ( 1 )
  {
    v8 = v5[4];
    if ( (__int64 *)v8 != v5 + 3 )
      break;
LABEL_8:
    v5 = (__int64 *)v5[1];
    if ( a3 + 40 == v5 )
      goto LABEL_9;
  }
  while ( 2 )
  {
    switch ( **(_WORD **)(v8 + 16) )
    {
      case 2:
      case 3:
      case 4:
      case 6:
      case 9:
      case 0xC:
      case 0xD:
      case 0x11:
      case 0x12:
        goto LABEL_6;
      default:
        if ( (*(_BYTE *)(v8 + 46) & 1) != 0 || !*(_QWORD *)(v8 + 64) )
        {
LABEL_6:
          if ( (*(_BYTE *)v8 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
              v8 = *(_QWORD *)(v8 + 8);
          }
          v8 = *(_QWORD *)(v8 + 8);
          if ( v5 + 3 == (__int64 *)v8 )
            goto LABEL_8;
          continue;
        }
        v19[0] = *(_QWORD *)(v8 + 64);
        sub_1623A60((__int64)v19, v19[0], 2);
        if ( !v19[0] )
        {
LABEL_9:
          *a1 = 0;
          return a1;
        }
        v10 = sub_1626D20(*a3);
        sub_3999410(a2, *(_QWORD *)(v10 + 8 * (5LL - *(unsigned int *)(v10 + 8))));
        v11 = sub_15C70A0((__int64)v19);
        while ( 1 )
        {
          v12 = v11;
          v13 = *(unsigned int *)(v11 + 8);
          if ( (_DWORD)v13 != 2 )
            break;
          v11 = *(_QWORD *)(v12 - 8);
          if ( !v11 )
          {
            v14 = -16;
            goto LABEL_20;
          }
        }
        v14 = -8 * v13;
LABEL_20:
        v15 = sub_15B1000(*(unsigned __int8 **)(v12 + v14));
        v18 = *(_QWORD *)(a2 + 4208);
        v16 = sub_398C0A0(a2);
        sub_3987590(*(_QWORD ***)(a2 + 8), *((_DWORD *)v15 + 7), 0, v15, 1u, a4, v16, v18);
        v17 = (unsigned __int8 *)v19[0];
        *a1 = v19[0];
        if ( v17 )
          sub_1623210((__int64)v19, v17, (__int64)a1);
        return a1;
    }
  }
}

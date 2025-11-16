// Function: sub_27DCA00
// Address: 0x27dca00
//
const void *__fastcall sub_27DCA00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // [rsp+8h] [rbp-38h]

  v8 = sub_AA54C0(a2);
  v9 = *(_BYTE *)a4;
  if ( *(_BYTE *)a4 <= 0x15u )
    return (const void *)a4;
  if ( v9 <= 0x1Cu )
    return sub_22CF3A0(*(__int64 **)(a1 + 32), a4, a3, v8, 0);
  v11 = *(_QWORD *)(a4 + 40);
  if ( v8 != v11 && a2 != v11 )
    return sub_22CF3A0(*(__int64 **)(a1 + 32), a4, a3, v8, 0);
  if ( v9 == 84 )
  {
    if ( v8 == v11 )
    {
      v13 = *(_QWORD *)(a4 - 8);
      v14 = 0x1FFFFFFFE0LL;
      if ( (*(_DWORD *)(a4 + 4) & 0x7FFFFFF) != 0 )
      {
        v15 = 0;
        do
        {
          if ( a3 == *(_QWORD *)(v13 + 32LL * *(unsigned int *)(a4 + 72) + 8 * v15) )
          {
            v14 = 32 * v15;
            goto LABEL_19;
          }
          ++v15;
        }
        while ( (*(_DWORD *)(a4 + 4) & 0x7FFFFFF) != (_DWORD)v15 );
        v14 = 0x1FFFFFFFE0LL;
      }
LABEL_19:
      a4 = *(_QWORD *)(v13 + v14);
      if ( *(_BYTE *)a4 <= 0x15u )
        return (const void *)a4;
    }
    return 0;
  }
  if ( (unsigned __int8)(v9 - 82) > 1u )
    return 0;
  if ( a2 != v11 )
    return 0;
  v16 = (_BYTE *)sub_27DCA00(a1, a2, a3, *(_QWORD *)(a4 - 64), a5);
  v12 = sub_27DCA00(a1, a2, a3, *(_QWORD *)(a4 - 32), a5);
  if ( !v16 || !v12 )
    return 0;
  return (const void *)sub_9719A0(*(_WORD *)(a4 + 2) & 0x3F, v16, v12, a5, 0, 0);
}

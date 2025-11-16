// Function: sub_3870AF0
// Address: 0x3870af0
//
__int64 __fastcall sub_3870AF0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rsi
  _BYTE *v3; // rdi
  int v4; // eax
  _BYTE *v5; // rcx
  __int64 v6; // r13
  _QWORD **v7; // rbx
  _QWORD **v8; // r13
  _QWORD *v9; // rsi
  _QWORD *v10; // rsi
  __int64 v12; // [rsp+0h] [rbp-100h] BYREF
  unsigned __int8 v13; // [rsp+8h] [rbp-F8h]
  __int64 *v14; // [rsp+10h] [rbp-F0h] BYREF
  _BYTE *v15; // [rsp+18h] [rbp-E8h]
  __int64 v16; // [rsp+20h] [rbp-E0h]
  _BYTE v17[64]; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v18; // [rsp+68h] [rbp-98h]
  _BYTE *v19; // [rsp+70h] [rbp-90h]
  _BYTE *v20; // [rsp+78h] [rbp-88h]
  __int64 v21; // [rsp+80h] [rbp-80h]
  int v22; // [rsp+88h] [rbp-78h]
  _BYTE v23[112]; // [rsp+90h] [rbp-70h] BYREF

  v14 = &v12;
  v12 = a2;
  v16 = 0x800000000LL;
  v19 = v23;
  v20 = v23;
  v13 = 0;
  v15 = v17;
  v18 = 0;
  v21 = 8;
  v22 = 0;
  sub_3870970((__int64 *)&v14, a1);
  v2 = (__int64)v14;
  v3 = v15;
  v4 = v16;
  while ( 1 )
  {
    v5 = &v3[8 * v4];
    if ( !v4 )
      break;
LABEL_3:
    if ( *(_BYTE *)(v2 + 8) )
      break;
    v6 = *((_QWORD *)v5 - 1);
    LODWORD(v16) = --v4;
    switch ( *(_WORD *)(v6 + 24) )
    {
      case 0:
      case 0xA:
        v5 -= 8;
        if ( v4 )
          goto LABEL_3;
        goto LABEL_12;
      case 1:
      case 2:
      case 3:
        v10 = *(_QWORD **)(v6 + 32);
        goto LABEL_9;
      case 4:
      case 5:
      case 7:
      case 8:
      case 9:
        v7 = *(_QWORD ***)(v6 + 32);
        v8 = &v7[*(_QWORD *)(v6 + 40)];
        if ( v7 == v8 )
          continue;
        do
        {
          v9 = *v7++;
          sub_3870970((__int64 *)&v14, v9);
        }
        while ( v8 != v7 );
        goto LABEL_10;
      case 6:
        sub_3870970((__int64 *)&v14, *(_QWORD **)(v6 + 32));
        v10 = *(_QWORD **)(v6 + 40);
LABEL_9:
        sub_3870970((__int64 *)&v14, v10);
LABEL_10:
        v2 = (__int64)v14;
        v3 = v15;
        v4 = v16;
        break;
    }
  }
LABEL_12:
  if ( v20 != v19 )
  {
    _libc_free((unsigned __int64)v20);
    v3 = v15;
  }
  if ( v3 != v17 )
    _libc_free((unsigned __int64)v3);
  return v13 ^ 1u;
}

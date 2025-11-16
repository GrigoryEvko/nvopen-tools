// Function: sub_3393F10
// Address: 0x3393f10
//
_QWORD *__fastcall sub_3393F10(__int64 a1, __int64 a2, int a3)
{
  int v3; // r9d
  __int64 v6; // r14
  int v7; // edx
  unsigned int v8; // ecx
  unsigned __int8 v9; // al
  char v10; // al
  int v11; // edx
  int v12; // edx
  int v13; // edx
  int v14; // edx
  int v15; // edx
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r14
  int v19; // r9d
  __int64 v20; // rdx
  __int64 v21; // r11
  __int64 v22; // r10
  unsigned __int16 *v23; // rax
  __int64 v24; // r8
  int v25; // ecx
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // r13
  int v29; // edx
  int v30; // r14d
  _QWORD *result; // rax
  bool v32; // al
  __int64 *v33; // rax
  int v34; // [rsp+8h] [rbp-78h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  int v37; // [rsp+24h] [rbp-5Ch]
  int v38; // [rsp+28h] [rbp-58h]
  int v39; // [rsp+28h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-40h] BYREF
  int v41; // [rsp+48h] [rbp-38h]

  v3 = 0;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    switch ( *(_BYTE *)a2 )
    {
      case ')':
      case '+':
      case '-':
      case '/':
      case '2':
      case '5':
      case 'J':
      case 'K':
      case 'S':
        goto LABEL_6;
      case 'T':
      case 'U':
      case 'V':
        v6 = *(_QWORD *)(a2 + 8);
        v7 = *(unsigned __int8 *)(v6 + 8);
        v8 = v7 - 17;
        v9 = *(_BYTE *)(v6 + 8);
        if ( (unsigned int)(v7 - 17) <= 1 )
          v9 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
        if ( v9 <= 3u || v9 == 5 || (v9 & 0xFD) == 4 )
          goto LABEL_6;
        if ( (_BYTE)v7 == 15 )
        {
          if ( (*(_BYTE *)(v6 + 9) & 4) == 0 )
            break;
          v32 = sub_BCB420(*(_QWORD *)(a2 + 8));
          v3 = 0;
          if ( !v32 )
            break;
          v33 = *(__int64 **)(v6 + 16);
          v6 = *v33;
          v7 = *(unsigned __int8 *)(*v33 + 8);
          v8 = v7 - 17;
        }
        else if ( (_BYTE)v7 == 16 )
        {
          do
          {
            v6 = *(_QWORD *)(v6 + 24);
            LOBYTE(v7) = *(_BYTE *)(v6 + 8);
          }
          while ( (_BYTE)v7 == 16 );
          v8 = (unsigned __int8)v7 - 17;
        }
        if ( v8 <= 1 )
          LOBYTE(v7) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
        if ( (unsigned __int8)v7 <= 3u || (_BYTE)v7 == 5 || (v7 & 0xFD) == 4 )
        {
LABEL_6:
          v10 = *(_BYTE *)(a2 + 1) >> 1;
          v3 = (16 * v10) & 0x20;
          if ( (v10 & 4) != 0 )
            v3 = (16 * v10) & 0x20 | 0x40;
          v11 = v3;
          if ( (v10 & 8) != 0 )
          {
            LOBYTE(v11) = v3 | 0x80;
            v3 = v11;
          }
          v12 = v3;
          if ( (v10 & 0x10) != 0 )
          {
            BYTE1(v12) = BYTE1(v3) | 1;
            v3 = v12;
          }
          v13 = v3;
          if ( (v10 & 0x20) != 0 )
          {
            BYTE1(v13) = BYTE1(v3) | 2;
            v3 = v13;
          }
          v14 = v3;
          if ( (v10 & 0x40) != 0 )
          {
            BYTE1(v14) = BYTE1(v3) | 4;
            v3 = v14;
          }
          v15 = v3;
          if ( (*(_BYTE *)(a2 + 1) & 2) != 0 )
          {
            BYTE1(v15) = BYTE1(v3) | 8;
            v3 = v15;
          }
        }
        break;
      default:
        break;
    }
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v16 = *(__int64 **)(a2 - 8);
  else
    v16 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v38 = v3;
  v17 = sub_338B750(a1, *v16);
  v18 = *(_QWORD *)(a1 + 864);
  v19 = v38;
  v21 = v20;
  v22 = v17;
  v23 = (unsigned __int16 *)(*(_QWORD *)(v17 + 48) + 16LL * (unsigned int)v20);
  LODWORD(v20) = *(_DWORD *)(a1 + 848);
  v24 = *((_QWORD *)v23 + 1);
  v25 = *v23;
  v40 = 0;
  v26 = *(_QWORD *)a1;
  v41 = v20;
  if ( v26 )
  {
    if ( &v40 != (__int64 *)(v26 + 48) )
    {
      v27 = *(_QWORD *)(v26 + 48);
      v40 = v27;
      if ( v27 )
      {
        v34 = v25;
        v35 = v22;
        v36 = v21;
        v37 = v38;
        v39 = v24;
        sub_B96E90((__int64)&v40, v27, 1);
        v25 = v34;
        v22 = v35;
        v21 = v36;
        v19 = v37;
        LODWORD(v24) = v39;
      }
    }
  }
  v28 = sub_33FA050(v18, a3, (unsigned int)&v40, v25, v24, v19, v22, v21);
  v30 = v29;
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
  v40 = a2;
  result = sub_337DC20(a1 + 8, &v40);
  *result = v28;
  *((_DWORD *)result + 2) = v30;
  return result;
}

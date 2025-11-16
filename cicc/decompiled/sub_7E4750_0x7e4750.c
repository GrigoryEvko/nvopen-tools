// Function: sub_7E4750
// Address: 0x7e4750
//
_QWORD *__fastcall sub_7E4750(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // r13
  char i; // al
  __int64 k; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 m; // r14
  unsigned __int64 v10; // r15
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rbx
  __int64 j; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // [rsp+8h] [rbp-48h]
  _BYTE *v20[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = a1;
  v20[0] = sub_724DC0();
  if ( (unsigned int)sub_7E1F40(a1) )
  {
    sub_72BAF0((__int64)v20[0], -1, unk_4D03F80);
    v2 = (_QWORD *)sub_724E50((__int64 *)v20, (_BYTE *)0xFFFFFFFFFFFFFFFFLL);
  }
  else
  {
    if ( !(unsigned int)sub_7E1F90(a1) )
      goto LABEL_9;
    v1 = sub_7E1D00();
    for ( i = *(_BYTE *)(v1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    {
      v1 = *(_QWORD *)(v1 + 160);
LABEL_9:
      ;
    }
    switch ( i )
    {
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 19:
        sub_72BB40(v1, (const __m128i *)v20[0]);
        v2 = (_QWORD *)sub_724E50((__int64 *)v20, v20[0]);
        goto LABEL_3;
      case 8:
        v14 = sub_8D4050(v1);
        v15 = *(_QWORD *)(v1 + 176);
        v16 = v14;
        v2 = sub_724D50(10);
        if ( v15 )
        {
          for ( j = 0; j != v15; ++j )
          {
            v18 = sub_7E4750(v16);
            if ( v2[22] )
              *(_QWORD *)(v2[23] + 120LL) = v18;
            else
              v2[22] = v18;
            v2[23] = v18;
          }
        }
        goto LABEL_25;
      case 9:
      case 10:
      case 11:
        v2 = sub_724D50(10);
        sub_7E3EE0(v1);
        v2[16] = v1;
        for ( k = *(_QWORD *)(v1 + 160); ; k = *(_QWORD *)(v7 + 112) )
        {
          v6 = sub_72FD90(k, 11);
          v7 = v6;
          if ( !v6 )
            break;
          v8 = sub_7E4750(*(_QWORD *)(v6 + 120));
          if ( v2[22] )
            *(_QWORD *)(v2[23] + 120LL) = v8;
          else
            v2[22] = v8;
          v2[23] = v8;
          if ( (unsigned int)sub_8D3B10(v1) )
            break;
        }
        goto LABEL_3;
      case 15:
        for ( m = *(_QWORD *)(v1 + 160); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
          ;
        v10 = *(_QWORD *)(m + 128);
        v19 = *(_QWORD *)(v1 + 128);
        v11 = v19 / v10;
        v2 = sub_724D50(10);
        if ( v19 < v10 )
          goto LABEL_25;
        v12 = 0;
        break;
      default:
        sub_721090();
    }
    do
    {
      while ( 1 )
      {
        v13 = sub_7E4750(m);
        if ( !v2[22] )
          break;
        ++v12;
        *(_QWORD *)(v2[23] + 120LL) = v13;
        v2[23] = v13;
        if ( v11 <= v12 )
          goto LABEL_25;
      }
      ++v12;
      v2[22] = v13;
      v2[23] = v13;
    }
    while ( v11 > v12 );
LABEL_25:
    v2[16] = v1;
  }
LABEL_3:
  if ( v20[0] )
    sub_724E30((__int64)v20);
  return v2;
}

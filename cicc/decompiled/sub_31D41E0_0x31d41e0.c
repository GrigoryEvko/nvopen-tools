// Function: sub_31D41E0
// Address: 0x31d41e0
//
__int64 __fastcall sub_31D41E0(__int64 a1)
{
  unsigned int v1; // eax
  char v2; // dl
  char v3; // r13
  unsigned int v4; // eax
  char v5; // dl
  unsigned int v6; // r12d
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 i; // r15
  unsigned int v10; // r13d
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // [rsp+8h] [rbp-58h]
  unsigned int v27; // [rsp+Ch] [rbp-54h]
  unsigned int v28; // [rsp+10h] [rbp-50h]

  v1 = sub_CE8FC0(a1);
  v3 = v2;
  v28 = v1;
  v4 = sub_CE8F50(a1);
  if ( v5 )
  {
    v6 = v4;
    if ( v3 )
      v6 = v28;
  }
  else
  {
    if ( !v3 )
      goto LABEL_5;
    v6 = v28;
  }
  v27 = v6;
  if ( v6 > 0x3FF )
  {
LABEL_5:
    v27 = 1024;
    v26 = 64;
    goto LABEL_6;
  }
  v14 = 64;
  if ( v6 <= 0x3F )
    v14 = v6;
  v26 = v14;
LABEL_6:
  v7 = a1 + 72;
  v8 = *(_QWORD *)(a1 + 80);
  if ( a1 + 72 == v8 )
  {
    return 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v8 + 32);
      if ( i != v8 + 24 )
        break;
      v8 = *(_QWORD *)(v8 + 8);
      if ( v7 == v8 )
        return 0;
      if ( !v8 )
        BUG();
    }
    v10 = 0;
    while ( v8 != v7 )
    {
      if ( !i )
        BUG();
      if ( *(_BYTE *)(i - 24) == 85 )
      {
        v12 = *(_QWORD *)(i - 56);
        if ( v12 )
        {
          if ( !*(_BYTE *)v12 && *(_QWORD *)(v12 + 24) == *(_QWORD *)(i + 56) && (*(_BYTE *)(v12 + 33) & 0x20) != 0 )
          {
            v13 = i - 24;
            switch ( *(_DWORD *)(v12 + 36) )
            {
              case 0x245B:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v23 = sub_B91C10(i - 24, 4), v13 = i - 24, !v23) )
                  v10 |= sub_31D3FC0(0, 0x7FFFFFFF, v13);
                break;
              case 0x245C:
              case 0x245D:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v19 = sub_B91C10(i - 24, 4), v13 = i - 24, !v19) )
                  v10 |= sub_31D3FC0(0, 0xFFFF, v13);
                break;
              case 0x2480:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v22 = sub_B91C10(i - 24, 4), v13 = i - 24, !v22) )
                  v10 |= sub_31D3FC0(0, 32, v13);
                break;
              case 0x248B:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v20 = sub_B91C10(i - 24, 4), v13 = i - 24, !v20) )
                  v10 |= sub_31D3FC0(1, 0x80000000LL, v13);
                break;
              case 0x248C:
              case 0x248D:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v18 = sub_B91C10(i - 24, 4), v13 = i - 24, !v18) )
                  v10 |= sub_31D3FC0(1, 0x10000, v13);
                break;
              case 0x2490:
              case 0x2491:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v15 = sub_B91C10(i - 24, 4), v13 = i - 24, !v15) )
                {
                  v16 = v27;
                  goto LABEL_37;
                }
                break;
              case 0x2492:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v25 = sub_B91C10(i - 24, 4), v13 = i - 24, !v25) )
                {
                  v16 = v26;
LABEL_37:
                  v10 |= sub_31D3FC0(1, v16 + 1, v13);
                }
                break;
              case 0x249A:
              case 0x249B:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v17 = sub_B91C10(i - 24, 4), v13 = i - 24, !v17) )
                  v10 |= sub_31D3FC0(0, v27, v13);
                break;
              case 0x249C:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v21 = sub_B91C10(i - 24, 4), v13 = i - 24, !v21) )
                  v10 |= sub_31D3FC0(0, v26, v13);
                break;
              case 0x249E:
                if ( (*(_BYTE *)(i - 17) & 0x20) == 0 || (v24 = sub_B91C10(i - 24, 4), v13 = i - 24, !v24) )
                  v10 |= sub_31D3FC0(32, 33, v13);
                break;
              default:
                break;
            }
          }
        }
      }
      for ( i = *(_QWORD *)(i + 8); i == v8 - 24 + 48; i = *(_QWORD *)(v8 + 32) )
      {
        v8 = *(_QWORD *)(v8 + 8);
        if ( v7 == v8 )
          return v10;
        if ( !v8 )
          BUG();
      }
    }
  }
  return v10;
}

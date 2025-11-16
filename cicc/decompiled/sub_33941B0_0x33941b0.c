// Function: sub_33941B0
// Address: 0x33941b0
//
_QWORD *__fastcall sub_33941B0(__int64 a1, unsigned __int8 *a2, int a3)
{
  unsigned __int64 v4; // rax
  int v5; // r13d
  int v6; // edx
  __int64 *v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r15
  unsigned __int8 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // r9
  unsigned __int16 *v17; // rax
  int v18; // ecx
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r13
  int v22; // edx
  int v23; // r14d
  _QWORD *result; // rax
  __int64 v25; // rcx
  int v26; // edx
  __int64 v27; // r14
  int v28; // edx
  unsigned int v29; // ecx
  unsigned __int8 v30; // al
  char v31; // al
  int v32; // edx
  int v33; // edx
  int v34; // edx
  int v35; // edx
  int v36; // edx
  __int64 *v37; // rax
  __int128 v38; // [rsp-20h] [rbp-B0h]
  __int128 v39; // [rsp-10h] [rbp-A0h]
  int v40; // [rsp+8h] [rbp-88h]
  __int64 v41; // [rsp+10h] [rbp-80h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  int v43; // [rsp+20h] [rbp-70h]
  __int64 v45; // [rsp+38h] [rbp-58h]
  unsigned __int8 *v46; // [rsp+50h] [rbp-40h] BYREF
  int v47; // [rsp+58h] [rbp-38h]

  v4 = *a2;
  if ( (unsigned __int8)v4 <= 0x1Cu )
  {
    v5 = 0;
    if ( (_BYTE)v4 != 5 )
      goto LABEL_12;
    v26 = *((unsigned __int16 *)a2 + 1);
    if ( (*((_WORD *)a2 + 1) & 0xFFF7) == 0x11 || (v26 & 0xFFFD) == 0xD )
    {
      if ( ((a2[1] >> 1) & 2) != 0 )
        v5 = ((a2[1] & 2) != 0) | 2;
      else
        v5 = (a2[1] & 2) != 0;
    }
    if ( (unsigned __int16)(v26 - 26) > 1u && (unsigned int)(v26 - 19) > 1 )
      goto LABEL_12;
    goto LABEL_6;
  }
  if ( (unsigned __int8)v4 <= 0x36u )
  {
    v25 = 0x40540000000000LL;
    v6 = (unsigned __int8)v4;
    if ( !_bittest64(&v25, v4) )
    {
      v5 = 0;
      if ( (unsigned int)(unsigned __int8)v4 - 48 > 1 )
      {
LABEL_10:
        switch ( v6 )
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
            goto LABEL_39;
          case 'T':
          case 'U':
          case 'V':
            v27 = *((_QWORD *)a2 + 1);
            v28 = *(unsigned __int8 *)(v27 + 8);
            v29 = v28 - 17;
            v30 = *(_BYTE *)(v27 + 8);
            if ( (unsigned int)(v28 - 17) <= 1 )
              v30 = *(_BYTE *)(**(_QWORD **)(v27 + 16) + 8LL);
            if ( v30 <= 3u || v30 == 5 || (v30 & 0xFD) == 4 )
              goto LABEL_39;
            if ( (_BYTE)v28 == 15 )
            {
              if ( (*(_BYTE *)(v27 + 9) & 4) == 0 || !sub_BCB420(*((_QWORD *)a2 + 1)) )
                goto LABEL_12;
              v37 = *(__int64 **)(v27 + 16);
              v27 = *v37;
              v28 = *(unsigned __int8 *)(*v37 + 8);
              v29 = v28 - 17;
            }
            else if ( (_BYTE)v28 == 16 )
            {
              do
              {
                v27 = *(_QWORD *)(v27 + 24);
                LOBYTE(v28) = *(_BYTE *)(v27 + 8);
              }
              while ( (_BYTE)v28 == 16 );
              v29 = (unsigned __int8)v28 - 17;
            }
            if ( v29 <= 1 )
              LOBYTE(v28) = *(_BYTE *)(**(_QWORD **)(v27 + 16) + 8LL);
            if ( (unsigned __int8)v28 <= 3u || (_BYTE)v28 == 5 || (v28 & 0xFD) == 4 )
            {
LABEL_39:
              v31 = a2[1] >> 1;
              if ( (v31 & 2) != 0 )
                v5 |= 0x20u;
              if ( (v31 & 4) != 0 )
                v5 |= 0x40u;
              v32 = v5;
              if ( (v31 & 8) != 0 )
              {
                LOBYTE(v32) = v5 | 0x80;
                v5 = v32;
              }
              v33 = v5;
              if ( (v31 & 0x10) != 0 )
              {
                BYTE1(v33) = BYTE1(v5) | 1;
                v5 = v33;
              }
              v34 = v5;
              if ( (v31 & 0x20) != 0 )
              {
                BYTE1(v34) = BYTE1(v5) | 2;
                v5 = v34;
              }
              v35 = v5;
              if ( (v31 & 0x40) != 0 )
              {
                BYTE1(v35) = BYTE1(v5) | 4;
                v5 = v35;
              }
              v36 = v5;
              if ( (a2[1] & 2) != 0 )
              {
                BYTE1(v36) = BYTE1(v5) | 8;
                v5 = v36;
              }
            }
            break;
          default:
            goto LABEL_12;
        }
        goto LABEL_12;
      }
      if ( (a2[1] & 2) == 0 )
        goto LABEL_9;
LABEL_7:
      v5 |= 4u;
LABEL_8:
      if ( (unsigned __int8)v4 <= 0x1Cu )
        goto LABEL_12;
      goto LABEL_9;
    }
    v5 = ((a2[1] & 2) != 0) | 2;
    if ( ((a2[1] >> 1) & 2) == 0 )
      v5 = (a2[1] & 2) != 0;
  }
  else
  {
    v5 = 0;
  }
  if ( (unsigned __int8)(v4 - 55) <= 1u || (unsigned int)(unsigned __int8)v4 - 48 <= 1 )
  {
LABEL_6:
    if ( (a2[1] & 2) == 0 )
      goto LABEL_8;
    goto LABEL_7;
  }
LABEL_9:
  v6 = (unsigned __int8)v4;
  if ( (_BYTE)v4 != 58 )
    goto LABEL_10;
  if ( (a2[1] & 2) != 0 )
    v5 |= 8u;
LABEL_12:
  if ( (a2[7] & 0x40) != 0 )
    v7 = (__int64 *)*((_QWORD *)a2 - 1);
  else
    v7 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v8 = sub_338B750(a1, *v7);
  v10 = v9;
  if ( (a2[7] & 0x40) != 0 )
    v11 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
  else
    v11 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v12 = sub_338B750(a1, *((_QWORD *)v11 + 4));
  v13 = *(_QWORD *)(a1 + 864);
  v14 = v12;
  v16 = v15;
  LODWORD(v15) = *(_DWORD *)(a1 + 848);
  v17 = (unsigned __int16 *)(*(_QWORD *)(v8 + 48) + 16LL * (unsigned int)v10);
  v45 = *((_QWORD *)v17 + 1);
  v18 = *v17;
  v19 = *(_QWORD *)a1;
  v46 = 0;
  v47 = v15;
  if ( v19 )
  {
    if ( &v46 != (unsigned __int8 **)(v19 + 48) )
    {
      v20 = *(_QWORD *)(v19 + 48);
      v46 = (unsigned __int8 *)v20;
      if ( v20 )
      {
        v40 = v18;
        v41 = v14;
        v42 = v16;
        v43 = v13;
        sub_B96E90((__int64)&v46, v20, 1);
        v18 = v40;
        v14 = v41;
        v16 = v42;
        LODWORD(v13) = v43;
      }
    }
  }
  *((_QWORD *)&v39 + 1) = v16;
  *(_QWORD *)&v39 = v14;
  *((_QWORD *)&v38 + 1) = v10;
  *(_QWORD *)&v38 = v8;
  v21 = sub_3405C90(v13, a3, (unsigned int)&v46, v18, v45, v5, v38, v39);
  v23 = v22;
  if ( v46 )
    sub_B91220((__int64)&v46, (__int64)v46);
  v46 = a2;
  result = sub_337DC20(a1 + 8, (__int64 *)&v46);
  *result = v21;
  *((_DWORD *)result + 2) = v23;
  return result;
}

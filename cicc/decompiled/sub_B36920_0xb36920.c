// Function: sub_B36920
// Address: 0xb36920
//
__int64 __fastcall sub_B36920(unsigned int **a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  unsigned __int64 v5; // r11
  int v6; // r10d
  __int64 v7; // rdx
  unsigned int *v8; // rcx
  unsigned int *v9; // r8
  unsigned int *v10; // rsi
  int v11; // edi
  __int64 v12; // rax
  unsigned int v13; // edi
  unsigned __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r13
  int v18; // edx
  unsigned int v19; // ecx
  unsigned __int8 v20; // al
  unsigned int *v21; // rdx
  unsigned int v22; // r13d
  __int64 v23; // r13
  unsigned int *v24; // rbx
  unsigned int *v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 *v29; // rax
  __int64 *v30; // rax
  __int64 v31; // [rsp+0h] [rbp-D0h]
  unsigned __int64 v32; // [rsp+8h] [rbp-C8h]
  int v33; // [rsp+10h] [rbp-C0h]
  __int64 v34; // [rsp+20h] [rbp-B0h]
  int v35; // [rsp+28h] [rbp-A8h]
  int v36; // [rsp+2Ch] [rbp-A4h]
  __int64 v37; // [rsp+38h] [rbp-98h] BYREF
  char v38[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v39; // [rsp+60h] [rbp-70h]
  _QWORD v40[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v41; // [rsp+90h] [rbp-40h]

  v3 = *(_QWORD *)(*((_QWORD *)a1[6] + 9) + 40LL);
  v40[0] = *(_QWORD *)(a2 + 8);
  v4 = sub_B6E160(v3, 208, v40, 1);
  v37 = a2;
  v5 = 0;
  v39 = 257;
  v6 = v4;
  if ( v4 )
    v5 = *(_QWORD *)(v4 + 24);
  v7 = (__int64)a1[15];
  v41 = 257;
  v8 = a1[14];
  v9 = &v8[14 * v7];
  if ( v9 == v8 )
  {
    v35 = 2;
    v13 = 2;
  }
  else
  {
    v10 = a1[14];
    v11 = 0;
    do
    {
      v12 = *((_QWORD *)v10 + 5) - *((_QWORD *)v10 + 4);
      v10 += 14;
      v11 += v12 >> 3;
    }
    while ( v9 != v10 );
    v13 = v11 + 2;
    v35 = v13 & 0x7FFFFFF;
  }
  v31 = (__int64)a1[14];
  v32 = v5;
  LOBYTE(v36) = 16 * (_DWORD)v7 != 0;
  v33 = v6;
  v14 = ((unsigned __int64)(unsigned int)(16 * v7) << 32) | v13;
  v34 = v7;
  v15 = sub_BD2CC0(88, v14);
  v16 = v15;
  if ( v15 )
  {
    sub_B44260(v15, **(_QWORD **)(v32 + 16), 56, v35 | (unsigned int)(v36 << 28), 0, 0);
    *(_QWORD *)(v16 + 72) = 0;
    v14 = v32;
    sub_B4A290(v16, v32, v33, (unsigned int)&v37, 1, (unsigned int)v40, v31, v34);
  }
  if ( *((_BYTE *)a1 + 108) )
  {
    v30 = (__int64 *)sub_BD5C60(v16, v14);
    *(_QWORD *)(v16 + 72) = sub_A7A090((__int64 *)(v16 + 72), v30, -1, 72);
  }
  if ( *(_BYTE *)v16 > 0x1Cu )
  {
    switch ( *(_BYTE *)v16 )
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
        goto LABEL_16;
      case 'T':
      case 'U':
      case 'V':
        v17 = *(_QWORD *)(v16 + 8);
        v18 = *(unsigned __int8 *)(v17 + 8);
        v19 = v18 - 17;
        v20 = *(_BYTE *)(v17 + 8);
        if ( (unsigned int)(v18 - 17) <= 1 )
          v20 = *(_BYTE *)(**(_QWORD **)(v17 + 16) + 8LL);
        if ( v20 <= 3u || v20 == 5 || (v20 & 0xFD) == 4 )
          goto LABEL_16;
        if ( (_BYTE)v18 == 15 )
        {
          if ( (*(_BYTE *)(v17 + 9) & 4) == 0 || !(unsigned __int8)sub_BCB420(*(_QWORD *)(v16 + 8)) )
            break;
          v29 = *(__int64 **)(v17 + 16);
          v17 = *v29;
          v18 = *(unsigned __int8 *)(*v29 + 8);
          v19 = v18 - 17;
        }
        else if ( (_BYTE)v18 == 16 )
        {
          do
          {
            v17 = *(_QWORD *)(v17 + 24);
            LOBYTE(v18) = *(_BYTE *)(v17 + 8);
          }
          while ( (_BYTE)v18 == 16 );
          v19 = (unsigned __int8)v18 - 17;
        }
        if ( v19 <= 1 )
          LOBYTE(v18) = *(_BYTE *)(**(_QWORD **)(v17 + 16) + 8LL);
        if ( (unsigned __int8)v18 <= 3u || (_BYTE)v18 == 5 || (v18 & 0xFD) == 4 )
        {
LABEL_16:
          v21 = a1[12];
          v22 = *((_DWORD *)a1 + 26);
          if ( v21 )
            sub_B99FD0(v16, 3, v21);
          sub_B45150(v16, v22);
        }
        break;
      default:
        break;
    }
  }
  (*(void (__fastcall **)(unsigned int *, __int64, char *, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v16,
    v38,
    a1[7],
    a1[8]);
  v23 = 4LL * *((unsigned int *)a1 + 2);
  v24 = *a1;
  v25 = &v24[v23];
  while ( v25 != v24 )
  {
    v26 = *((_QWORD *)v24 + 1);
    v27 = *v24;
    v24 += 4;
    sub_B99FD0(v16, v27, v26);
  }
  return v16;
}

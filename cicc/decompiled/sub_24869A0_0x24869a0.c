// Function: sub_24869A0
// Address: 0x24869a0
//
void __fastcall sub_24869A0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6, __int64 a7)
{
  int v7; // r12d
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v14; // rcx
  int v15; // esi
  __int64 v16; // rax
  _QWORD *v17; // rbx
  __int64 v18; // r12
  int v19; // edx
  unsigned int v20; // ecx
  unsigned __int8 v21; // al
  int v22; // r12d
  __int64 v23; // r12
  __int64 v24; // r13
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 v29; // [rsp+8h] [rbp-98h]
  __int64 v30; // [rsp+10h] [rbp-90h]
  unsigned int v31; // [rsp+24h] [rbp-7Ch]
  char v35[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v36; // [rsp+60h] [rbp-40h]

  v11 = a1[15];
  v12 = a1[14];
  v36 = 257;
  v13 = v12 + 56 * v11;
  if ( v13 == v12 )
  {
    v15 = 0;
  }
  else
  {
    v14 = v12;
    v15 = 0;
    do
    {
      v16 = *(_QWORD *)(v14 + 40) - *(_QWORD *)(v14 + 32);
      v14 += 56;
      v15 += v16 >> 3;
    }
    while ( v13 != v14 );
  }
  v29 = v12;
  v30 = v11;
  LOBYTE(v7) = 16 * (_DWORD)v11 != 0;
  v31 = v15 + a5 + 1;
  v17 = sub_BD2CC0(88, ((unsigned __int64)(unsigned int)(16 * v11) << 32) | v31);
  if ( v17 )
  {
    sub_B44260((__int64)v17, **(_QWORD **)(a2 + 16), 56, (v7 << 28) | v31 & 0x7FFFFFF, 0, 0);
    v17[9] = 0;
    sub_B4A290((__int64)v17, a2, a3, a4, a5, (__int64)v35, v29, v30);
  }
  if ( *((_BYTE *)a1 + 108) )
  {
    v28 = (__int64 *)sub_BD5C60((__int64)v17);
    v17[9] = sub_A7A090(v17 + 9, v28, -1, 72);
  }
  if ( *(_BYTE *)v17 > 0x1Cu )
  {
    switch ( *(_BYTE *)v17 )
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
        goto LABEL_13;
      case 'T':
      case 'U':
      case 'V':
        v18 = v17[1];
        v19 = *(unsigned __int8 *)(v18 + 8);
        v20 = v19 - 17;
        v21 = *(_BYTE *)(v18 + 8);
        if ( (unsigned int)(v19 - 17) <= 1 )
          v21 = *(_BYTE *)(**(_QWORD **)(v18 + 16) + 8LL);
        if ( v21 <= 3u || v21 == 5 || (v21 & 0xFD) == 4 )
          goto LABEL_13;
        if ( (_BYTE)v19 == 15 )
        {
          if ( (*(_BYTE *)(v18 + 9) & 4) == 0 || !sub_BCB420(v17[1]) )
            break;
          v27 = *(__int64 **)(v18 + 16);
          v18 = *v27;
          v19 = *(unsigned __int8 *)(*v27 + 8);
          v20 = v19 - 17;
        }
        else if ( (_BYTE)v19 == 16 )
        {
          do
          {
            v18 = *(_QWORD *)(v18 + 24);
            LOBYTE(v19) = *(_BYTE *)(v18 + 8);
          }
          while ( (_BYTE)v19 == 16 );
          v20 = (unsigned __int8)v19 - 17;
        }
        if ( v20 <= 1 )
          LOBYTE(v19) = *(_BYTE *)(**(_QWORD **)(v18 + 16) + 8LL);
        if ( (unsigned __int8)v19 <= 3u || (_BYTE)v19 == 5 || (v19 & 0xFD) == 4 )
        {
LABEL_13:
          v22 = *((_DWORD *)a1 + 26);
          if ( a7 || (a7 = a1[12]) != 0 )
            sub_B99FD0((__int64)v17, 3u, a7);
          sub_B45150((__int64)v17, v22);
        }
        break;
      default:
        break;
    }
  }
  (*(void (__fastcall **)(__int64, _QWORD *, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v17,
    a6,
    a1[7],
    a1[8]);
  v23 = *a1;
  v24 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v24 )
  {
    do
    {
      v25 = *(_QWORD *)(v23 + 8);
      v26 = *(_DWORD *)v23;
      v23 += 16;
      sub_B99FD0((__int64)v17, v26, v25);
    }
    while ( v24 != v23 );
  }
}

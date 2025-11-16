// Function: sub_921880
// Address: 0x921880
//
__int64 __fastcall sub_921880(
        unsigned int **a1,
        unsigned __int64 a2,
        int a3,
        int a4,
        int a5,
        __int64 a6,
        unsigned int *a7)
{
  int v7; // ebx
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v14; // rcx
  int v15; // esi
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v20; // rbx
  int v21; // edx
  unsigned int v22; // ecx
  unsigned __int8 v23; // al
  unsigned int v24; // r13d
  unsigned int *v25; // rbx
  __int64 v26; // r13
  __int64 v27; // rdx
  __int64 v28; // rsi
  __int64 *v30; // rax
  __int64 v31; // rax
  __int64 v32; // [rsp-8h] [rbp-A8h]
  __int64 v33; // [rsp+8h] [rbp-98h]
  __int64 v34; // [rsp+10h] [rbp-90h]
  unsigned int v35; // [rsp+24h] [rbp-7Ch]
  char v39; // [rsp+40h] [rbp-60h] BYREF
  __int16 v40; // [rsp+60h] [rbp-40h]

  v11 = (__int64)a1[15];
  v12 = (__int64)a1[14];
  v40 = 257;
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
  v33 = v12;
  v34 = v11;
  LOBYTE(v7) = 16 * (_DWORD)v11 != 0;
  v35 = v15 + a5 + 1;
  v17 = ((unsigned __int64)(unsigned int)(16 * v11) << 32) | v35;
  v19 = sub_BD2CC0(88, v17);
  if ( v19 )
  {
    sub_B44260(v19, **(_QWORD **)(a2 + 16), 56, (v7 << 28) | v35 & 0x7FFFFFF, 0, 0);
    v17 = a2;
    *(_QWORD *)(v19 + 72) = 0;
    sub_B4A290(v19, a2, a3, a4, a5, (unsigned int)&v39, v33, v34);
    v18 = v32;
  }
  if ( *((_BYTE *)a1 + 108) )
  {
    v31 = sub_BD5C60(v19, v17, v18);
    *(_QWORD *)(v19 + 72) = sub_A7A090(v19 + 72, v31, 0xFFFFFFFFLL, 72);
  }
  if ( *(_BYTE *)v19 > 0x1Cu )
  {
    switch ( *(_BYTE *)v19 )
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
        v20 = *(_QWORD *)(v19 + 8);
        v21 = *(unsigned __int8 *)(v20 + 8);
        v22 = v21 - 17;
        v23 = *(_BYTE *)(v20 + 8);
        if ( (unsigned int)(v21 - 17) <= 1 )
          v23 = *(_BYTE *)(**(_QWORD **)(v20 + 16) + 8LL);
        if ( v23 <= 3u || v23 == 5 || (v23 & 0xFD) == 4 )
          goto LABEL_13;
        if ( (_BYTE)v21 == 15 )
        {
          if ( (*(_BYTE *)(v20 + 9) & 4) == 0 || !(unsigned __int8)sub_BCB420(*(_QWORD *)(v19 + 8)) )
            break;
          v30 = *(__int64 **)(v20 + 16);
          v20 = *v30;
          v21 = *(unsigned __int8 *)(*v30 + 8);
          v22 = v21 - 17;
        }
        else if ( (_BYTE)v21 == 16 )
        {
          do
          {
            v20 = *(_QWORD *)(v20 + 24);
            LOBYTE(v21) = *(_BYTE *)(v20 + 8);
          }
          while ( (_BYTE)v21 == 16 );
          v22 = (unsigned __int8)v21 - 17;
        }
        if ( v22 <= 1 )
          LOBYTE(v21) = *(_BYTE *)(**(_QWORD **)(v20 + 16) + 8LL);
        if ( (unsigned __int8)v21 <= 3u || (_BYTE)v21 == 5 || (v21 & 0xFD) == 4 )
        {
LABEL_13:
          v24 = *((_DWORD *)a1 + 26);
          if ( a7 || (a7 = a1[12]) != 0 )
            sub_B99FD0(v19, 3, a7);
          sub_B45150(v19, v24);
        }
        break;
      default:
        break;
    }
  }
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v19,
    a6,
    a1[7],
    a1[8]);
  v25 = *a1;
  v26 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v26 )
  {
    do
    {
      v27 = *((_QWORD *)v25 + 1);
      v28 = *v25;
      v25 += 4;
      sub_B99FD0(v19, v28, v27);
    }
    while ( (unsigned int *)v26 != v25 );
  }
  return v19;
}

// Function: sub_D5C860
// Address: 0xd5c860
//
__int64 __fastcall sub_D5C860(__int64 *a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rbx
  int v9; // edx
  unsigned int v10; // ecx
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  int v13; // r15d
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 *v19; // rax
  const char *v20; // [rsp+0h] [rbp-60h] BYREF
  __int16 v21; // [rsp+20h] [rbp-40h]

  v21 = 257;
  v6 = sub_BD2DA0(80);
  v7 = v6;
  if ( v6 )
  {
    sub_B44260(v6, a2, 55, 0x8000000u, 0, 0);
    *(_DWORD *)(v7 + 72) = a3;
    sub_BD6B50((unsigned __int8 *)v7, &v20);
    sub_BD2A10(v7, *(_DWORD *)(v7 + 72), 1);
  }
  if ( *(_BYTE *)v7 > 0x1Cu )
  {
    switch ( *(_BYTE *)v7 )
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
        goto LABEL_8;
      case 'T':
      case 'U':
      case 'V':
        v8 = *(_QWORD *)(v7 + 8);
        v9 = *(unsigned __int8 *)(v8 + 8);
        v10 = v9 - 17;
        v11 = *(_BYTE *)(v8 + 8);
        if ( (unsigned int)(v9 - 17) <= 1 )
          v11 = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
        if ( v11 <= 3u || v11 == 5 || (v11 & 0xFD) == 4 )
          goto LABEL_8;
        if ( (_BYTE)v9 == 15 )
        {
          if ( (*(_BYTE *)(v8 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v7 + 8)) )
            break;
          v19 = *(__int64 **)(v8 + 16);
          v8 = *v19;
          v9 = *(unsigned __int8 *)(*v19 + 8);
          v10 = v9 - 17;
        }
        else if ( (_BYTE)v9 == 16 )
        {
          do
          {
            v8 = *(_QWORD *)(v8 + 24);
            LOBYTE(v9) = *(_BYTE *)(v8 + 8);
          }
          while ( (_BYTE)v9 == 16 );
          v10 = (unsigned __int8)v9 - 17;
        }
        if ( v10 <= 1 )
          LOBYTE(v9) = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
        if ( (unsigned __int8)v9 <= 3u || (_BYTE)v9 == 5 || (v9 & 0xFD) == 4 )
        {
LABEL_8:
          v12 = a1[12];
          v13 = *((_DWORD *)a1 + 26);
          if ( v12 )
            sub_B99FD0(v7, 3u, v12);
          sub_B45150(v7, v13);
        }
        break;
      default:
        break;
    }
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v7,
    a4,
    a1[7],
    a1[8]);
  v14 = *a1;
  v15 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v15 )
  {
    do
    {
      v16 = *(_QWORD *)(v14 + 8);
      v17 = *(_DWORD *)v14;
      v14 += 16;
      sub_B99FD0(v7, v17, v16);
    }
    while ( v15 != v14 );
  }
  return v7;
}

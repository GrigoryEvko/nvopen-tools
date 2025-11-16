// Function: sub_36F1A10
// Address: 0x36f1a10
//
unsigned __int64 __fastcall sub_36F1A10(__int64 *a1, unsigned __int64 a2, __int64 **a3, __int64 a4)
{
  __int64 v5; // rdi
  __int64 (__fastcall *v8)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v9; // r12
  __int64 v11; // r13
  int v12; // edx
  unsigned int v13; // ecx
  unsigned __int8 v14; // al
  __int64 v15; // rdx
  int v16; // r13d
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 *v21; // rax
  char v22[32]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v23; // [rsp+20h] [rbp-40h]

  if ( a3 == *(__int64 ***)(a2 + 8) )
    return a2;
  v5 = a1[10];
  v8 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v5 + 120LL);
  if ( v8 != sub_920130 )
  {
    v9 = v8(v5, 50u, (_BYTE *)a2, (__int64)a3);
    goto LABEL_6;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x32u) )
      v9 = sub_ADAB70(50, a2, a3, 0);
    else
      v9 = sub_AA93C0(0x32u, a2, (__int64)a3);
LABEL_6:
    if ( v9 )
      return v9;
  }
  v23 = 257;
  v9 = sub_B51D30(50, a2, (__int64)a3, (__int64)v22, 0, 0);
  if ( *(_BYTE *)v9 > 0x1Cu )
  {
    switch ( *(_BYTE *)v9 )
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
        v11 = *(_QWORD *)(v9 + 8);
        v12 = *(unsigned __int8 *)(v11 + 8);
        v13 = v12 - 17;
        v14 = *(_BYTE *)(v11 + 8);
        if ( (unsigned int)(v12 - 17) <= 1 )
          v14 = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
        if ( v14 <= 3u || v14 == 5 || (v14 & 0xFD) == 4 )
          goto LABEL_13;
        if ( (_BYTE)v12 == 15 )
        {
          if ( (*(_BYTE *)(v11 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v9 + 8)) )
            break;
          v21 = *(__int64 **)(v11 + 16);
          v11 = *v21;
          v12 = *(unsigned __int8 *)(*v21 + 8);
          v13 = v12 - 17;
        }
        else if ( (_BYTE)v12 == 16 )
        {
          do
          {
            v11 = *(_QWORD *)(v11 + 24);
            LOBYTE(v12) = *(_BYTE *)(v11 + 8);
          }
          while ( (_BYTE)v12 == 16 );
          v13 = (unsigned __int8)v12 - 17;
        }
        if ( v13 <= 1 )
          LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
        if ( (unsigned __int8)v12 <= 3u || (_BYTE)v12 == 5 || (v12 & 0xFD) == 4 )
        {
LABEL_13:
          v15 = a1[12];
          v16 = *((_DWORD *)a1 + 26);
          if ( v15 )
            sub_B99FD0(v9, 3u, v15);
          sub_B45150(v9, v16);
        }
        break;
      default:
        break;
    }
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    a4,
    a1[7],
    a1[8]);
  v17 = *a1;
  v18 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v18 )
  {
    do
    {
      v19 = *(_QWORD *)(v17 + 8);
      v20 = *(_DWORD *)v17;
      v17 += 16;
      sub_B99FD0(v9, v20, v19);
    }
    while ( v18 != v17 );
  }
  return v9;
}

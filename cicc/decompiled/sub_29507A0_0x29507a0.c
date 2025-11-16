// Function: sub_29507A0
// Address: 0x29507a0
//
__int64 __fastcall sub_29507A0(__int64 *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, int a7, char a8)
{
  __int64 v10; // r12
  __int64 v12; // r13
  int v13; // edx
  unsigned int v14; // ecx
  unsigned __int8 v15; // al
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 *v21; // rax
  _BYTE v24[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v25; // [rsp+30h] [rbp-40h]

  if ( *(_QWORD *)(a3 + 8) == a4 )
    return a3;
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1[10] + 120LL))(a1[10]);
  if ( !v10 )
  {
    v25 = 257;
    v10 = sub_B51D30(a2, a3, a4, (__int64)v24, 0, 0);
    if ( *(_BYTE *)v10 > 0x1Cu )
    {
      switch ( *(_BYTE *)v10 )
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
          goto LABEL_9;
        case 'T':
        case 'U':
        case 'V':
          v12 = *(_QWORD *)(v10 + 8);
          v13 = *(unsigned __int8 *)(v12 + 8);
          v14 = v13 - 17;
          v15 = *(_BYTE *)(v12 + 8);
          if ( (unsigned int)(v13 - 17) <= 1 )
            v15 = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
          if ( v15 <= 3u || v15 == 5 || (v15 & 0xFD) == 4 )
            goto LABEL_9;
          if ( (_BYTE)v13 == 15 )
          {
            if ( (*(_BYTE *)(v12 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v10 + 8)) )
              break;
            v21 = *(__int64 **)(v12 + 16);
            v12 = *v21;
            v13 = *(unsigned __int8 *)(*v21 + 8);
            v14 = v13 - 17;
          }
          else if ( (_BYTE)v13 == 16 )
          {
            do
            {
              v12 = *(_QWORD *)(v12 + 24);
              LOBYTE(v13) = *(_BYTE *)(v12 + 8);
            }
            while ( (_BYTE)v13 == 16 );
            v14 = (unsigned __int8)v13 - 17;
          }
          if ( v14 <= 1 )
            LOBYTE(v13) = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
          if ( (unsigned __int8)v13 <= 3u || (_BYTE)v13 == 5 || (v13 & 0xFD) == 4 )
          {
LABEL_9:
            if ( !a8 )
              a7 = *((_DWORD *)a1 + 26);
            if ( a6 || (a6 = a1[12]) != 0 )
              sub_B99FD0(v10, 3u, a6);
            sub_B45150(v10, a7);
          }
          break;
        default:
          break;
      }
    }
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v10,
      a5,
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
        sub_B99FD0(v10, v20, v19);
      }
      while ( v18 != v17 );
    }
  }
  return v10;
}

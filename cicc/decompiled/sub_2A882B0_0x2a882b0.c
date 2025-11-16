// Function: sub_2A882B0
// Address: 0x2a882b0
//
unsigned __int64 __fastcall sub_2A882B0(
        __int64 *a1,
        unsigned int a2,
        unsigned __int64 a3,
        __int64 **a4,
        __int64 a5,
        __int64 a6,
        int a7,
        char a8)
{
  __int64 (*v10)(void); // rax
  __int64 v11; // r12
  __int64 v13; // r13
  int v14; // edx
  unsigned int v15; // ecx
  unsigned __int8 v16; // al
  __int64 v18; // rbx
  __int64 v19; // r13
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 *v22; // rax
  _BYTE v25[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v26; // [rsp+30h] [rbp-40h]

  if ( *(__int64 ***)(a3 + 8) == a4 )
    return a3;
  v10 = *(__int64 (**)(void))(*(_QWORD *)a1[10] + 120LL);
  if ( (char *)v10 != (char *)sub_920130 )
  {
    v11 = v10();
    goto LABEL_6;
  }
  if ( *(_BYTE *)a3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(a2) )
      v11 = sub_ADAB70(a2, a3, a4, 0);
    else
      v11 = sub_AA93C0(a2, a3, (__int64)a4);
LABEL_6:
    if ( v11 )
      return v11;
  }
  v26 = 257;
  v11 = sub_B51D30(a2, a3, (__int64)a4, (__int64)v25, 0, 0);
  if ( *(_BYTE *)v11 > 0x1Cu )
  {
    switch ( *(_BYTE *)v11 )
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
        v13 = *(_QWORD *)(v11 + 8);
        v14 = *(unsigned __int8 *)(v13 + 8);
        v15 = v14 - 17;
        v16 = *(_BYTE *)(v13 + 8);
        if ( (unsigned int)(v14 - 17) <= 1 )
          v16 = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
        if ( v16 <= 3u || v16 == 5 || (v16 & 0xFD) == 4 )
          goto LABEL_13;
        if ( (_BYTE)v14 == 15 )
        {
          if ( (*(_BYTE *)(v13 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v11 + 8)) )
            break;
          v22 = *(__int64 **)(v13 + 16);
          v13 = *v22;
          v14 = *(unsigned __int8 *)(*v22 + 8);
          v15 = v14 - 17;
        }
        else if ( (_BYTE)v14 == 16 )
        {
          do
          {
            v13 = *(_QWORD *)(v13 + 24);
            LOBYTE(v14) = *(_BYTE *)(v13 + 8);
          }
          while ( (_BYTE)v14 == 16 );
          v15 = (unsigned __int8)v14 - 17;
        }
        if ( v15 <= 1 )
          LOBYTE(v14) = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
        if ( (unsigned __int8)v14 <= 3u || (_BYTE)v14 == 5 || (v14 & 0xFD) == 4 )
        {
LABEL_13:
          if ( !a8 )
            a7 = *((_DWORD *)a1 + 26);
          if ( a6 || (a6 = a1[12]) != 0 )
            sub_B99FD0(v11, 3u, a6);
          sub_B45150(v11, a7);
        }
        break;
      default:
        break;
    }
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v11,
    a5,
    a1[7],
    a1[8]);
  v18 = *a1;
  v19 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v19 )
  {
    do
    {
      v20 = *(_QWORD *)(v18 + 8);
      v21 = *(_DWORD *)v18;
      v18 += 16;
      sub_B99FD0(v11, v21, v20);
    }
    while ( v19 != v18 );
  }
  return v11;
}

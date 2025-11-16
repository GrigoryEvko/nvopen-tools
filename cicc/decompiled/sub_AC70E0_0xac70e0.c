// Function: sub_AC70E0
// Address: 0xac70e0
//
__int64 __fastcall sub_AC70E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v8; // rdi
  int v9; // edx
  __int16 v10; // ax
  __int64 v11; // rdi

  switch ( *(_BYTE *)a1 )
  {
    case 4:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xB:
    case 0xC:
    case 0xD:
    case 0xE:
    case 0x14:
    case 0x15:
      goto LABEL_10;
    case 5:
      v9 = *(unsigned __int16 *)(a1 + 2);
      v10 = *(_WORD *)(a1 + 2);
      if ( (unsigned int)(v9 - 38) <= 0xC || (unsigned int)(v9 - 13) <= 0x11 || v10 == 61 || v10 == 62 )
        goto LABEL_10;
      if ( v10 == 63 )
      {
        v11 = *(_QWORD *)(a1 + 24);
        if ( v11 != a1 + 40 )
          _libc_free(v11, a2);
      }
      else
      {
        if ( v10 != 34 )
LABEL_25:
          BUG();
        if ( *(_BYTE *)(a1 + 72) )
          sub_9963D0(a1 + 40);
      }
LABEL_10:
      sub_BD7260(a1);
      return sub_BD2DD0(a1);
    case 0xF:
    case 0x10:
      v3 = *(_QWORD *)(a1 + 32);
      if ( v3 )
      {
        v4 = *(_QWORD *)(v3 + 32);
        if ( v4 )
        {
          v5 = *(_QWORD *)(v4 + 32);
          if ( v5 )
          {
            v6 = *(_QWORD *)(v5 + 32);
            if ( v6 )
            {
              sub_AC5B80((__int64 *)(v6 + 32));
              sub_BD7260(v6);
              sub_BD2DD0(v6);
            }
            sub_BD7260(v5);
            sub_BD2DD0(v5);
          }
          sub_BD7260(v4);
          sub_BD2DD0(v4);
        }
        sub_BD7260(v3);
        sub_BD2DD0(v3);
      }
      goto LABEL_10;
    case 0x11:
      if ( *(_DWORD *)(a1 + 32) > 0x40u )
      {
        v8 = *(_QWORD *)(a1 + 24);
        if ( v8 )
          j_j___libc_free_0_0(v8);
      }
      goto LABEL_10;
    case 0x12:
      sub_91D830((_QWORD *)(a1 + 24));
      goto LABEL_10;
    default:
      goto LABEL_25;
  }
}

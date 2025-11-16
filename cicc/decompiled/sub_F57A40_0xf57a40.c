// Function: sub_F57A40
// Address: 0xf57a40
//
void __fastcall sub_F57A40(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  bool v4; // zf
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned int *v7; // rdi
  unsigned int *v8; // rbx
  unsigned int *v9; // r14
  __int64 v10; // kr00_8
  __int64 v11; // [rsp+10h] [rbp-D0h]
  __int64 v12; // [rsp+18h] [rbp-C8h]
  unsigned int *v13; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v14; // [rsp+28h] [rbp-B8h]
  _BYTE v15[176]; // [rsp+30h] [rbp-B0h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a2 + 48) == 0;
  v13 = (unsigned int *)v15;
  v14 = 0x800000000LL;
  if ( !v4 || (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    a2 = (__int64)&v13;
    sub_B9AA80(v2, (__int64)&v13);
  }
  sub_BD5C60(a1);
  v11 = *(_QWORD *)(a1 + 8);
  v5 = sub_B43CC0(v2);
  v7 = v13;
  v12 = v5;
  v8 = &v13[4 * (unsigned int)v14];
  if ( v8 != v13 )
  {
    v9 = v13;
    do
    {
      a2 = *v9;
      v10 = v6;
      v6 = (unsigned int)a2;
      switch ( (int)a2 )
      {
        case 0:
        case 1:
        case 2:
        case 3:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 25:
        case 29:
        case 41:
          goto LABEL_7;
        case 4:
          a2 = v2;
          sub_F57900(v12, v2, *((_QWORD *)v9 + 1), a1);
          break;
        case 11:
          a2 = *((_QWORD *)v9 + 1);
          sub_F57830(v2, a2, a1);
          break;
        case 12:
        case 13:
        case 17:
          if ( *(_BYTE *)(v11 + 8) == 14 )
LABEL_7:
            sub_B99FD0(a1, a2, *((_QWORD *)v9 + 1));
          break;
        default:
          v6 = v10;
          break;
      }
      v9 += 4;
    }
    while ( v8 != v9 );
    v7 = v13;
  }
  if ( v7 != (unsigned int *)v15 )
    _libc_free(v7, a2);
}

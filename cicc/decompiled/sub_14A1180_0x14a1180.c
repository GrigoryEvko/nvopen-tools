// Function: sub_14A1180
// Address: 0x14a1180
//
__int64 __fastcall sub_14A1180(__int64 a1, unsigned int a2, __int64 a3, __int64 **a4, unsigned __int64 a5)
{
  int v5; // r13d
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r13
  unsigned int v10; // r12d
  _BYTE *v12; // [rsp+10h] [rbp-80h] BYREF
  __int64 v13; // [rsp+18h] [rbp-78h]
  _BYTE v14[112]; // [rsp+20h] [rbp-70h] BYREF

  v5 = a5;
  v12 = v14;
  v13 = 0x800000000LL;
  if ( a5 > 8 )
    sub_16CD150(&v12, v14, a5, 8);
  if ( v5 )
  {
    v7 = (unsigned int)v13;
    v8 = (__int64)&a4[(unsigned int)(v5 - 1) + 1];
    do
    {
      v9 = **a4;
      if ( HIDWORD(v13) <= (unsigned int)v7 )
      {
        sub_16CD150(&v12, v14, 0, 8);
        v7 = (unsigned int)v13;
      }
      ++a4;
      *(_QWORD *)&v12[8 * v7] = v9;
      v7 = (unsigned int)(v13 + 1);
      LODWORD(v13) = v13 + 1;
    }
    while ( (__int64 **)v8 != a4 );
  }
  if ( a2 > 0x1189 )
    goto LABEL_13;
  if ( a2 > 0x1182 )
  {
    v10 = ((1LL << ((unsigned __int8)a2 + 125)) & 0x49) == 0 ? 1 : 4;
  }
  else
  {
    if ( a2 <= 0x95 )
    {
      if ( a2 > 2 )
      {
        switch ( a2 )
        {
          case 3u:
          case 4u:
          case 0xEu:
          case 0xFu:
          case 0x12u:
          case 0x13u:
          case 0x14u:
          case 0x17u:
          case 0x1Bu:
          case 0x1Cu:
          case 0x1Du:
          case 0x24u:
          case 0x25u:
          case 0x26u:
          case 0x4Cu:
          case 0x4Du:
          case 0x71u:
          case 0x72u:
          case 0x74u:
          case 0x75u:
          case 0x90u:
          case 0x95u:
            goto LABEL_20;
          default:
            break;
        }
      }
LABEL_13:
      v10 = 1;
      goto LABEL_14;
    }
    if ( a2 == 191 )
LABEL_20:
      v10 = 0;
    else
      v10 = a2 != 215;
  }
LABEL_14:
  if ( v12 != v14 )
    _libc_free((unsigned __int64)v12);
  return v10;
}

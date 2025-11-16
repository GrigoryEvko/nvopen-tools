// Function: sub_39B1B80
// Address: 0x39b1b80
//
__int64 __fastcall sub_39B1B80(__int64 a1, unsigned int a2, __int64 a3, __int64 **a4, unsigned __int64 a5)
{
  _BYTE *v5; // r9
  int v6; // r13d
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r13
  __int64 (*v11)(); // rax
  unsigned int v12; // r12d
  char v13; // al
  _BYTE *v15; // [rsp+8h] [rbp-88h]
  _BYTE *v16; // [rsp+8h] [rbp-88h]
  _BYTE *v17; // [rsp+10h] [rbp-80h] BYREF
  __int64 v18; // [rsp+18h] [rbp-78h]
  _BYTE v19[112]; // [rsp+20h] [rbp-70h] BYREF

  v5 = v19;
  v6 = a5;
  v17 = v19;
  v18 = 0x800000000LL;
  if ( a5 > 8 )
  {
    sub_16CD150((__int64)&v17, v19, a5, 8, a5, (int)v19);
    v5 = v19;
  }
  if ( v6 )
  {
    v8 = (unsigned int)v18;
    v9 = (__int64)&a4[(unsigned int)(v6 - 1) + 1];
    do
    {
      v10 = **a4;
      if ( HIDWORD(v18) <= (unsigned int)v8 )
      {
        v15 = v5;
        sub_16CD150((__int64)&v17, v5, 0, 8, a5, (int)v5);
        v8 = (unsigned int)v18;
        v5 = v15;
      }
      ++a4;
      *(_QWORD *)&v17[8 * v8] = v10;
      v8 = (unsigned int)(v18 + 1);
      LODWORD(v18) = v18 + 1;
    }
    while ( a4 != (__int64 **)v9 );
  }
  if ( a2 == 33 )
  {
    v11 = *(__int64 (**)())(**(_QWORD **)(a1 + 24) + 152LL);
    if ( v11 == sub_1D5A370 )
      goto LABEL_16;
LABEL_18:
    v16 = v5;
    v13 = v11();
    v5 = v16;
    if ( v13 )
      goto LABEL_19;
LABEL_16:
    v12 = 4;
    goto LABEL_20;
  }
  if ( a2 == 31 )
  {
    v11 = *(__int64 (**)())(**(_QWORD **)(a1 + 24) + 160LL);
    if ( v11 == sub_1D5A380 )
      goto LABEL_16;
    goto LABEL_18;
  }
  if ( a2 > 0x1189 )
  {
LABEL_19:
    v12 = 1;
    goto LABEL_20;
  }
  if ( a2 > 0x1182 )
  {
    v12 = ((1LL << ((unsigned __int8)a2 + 125)) & 0x49) == 0 ? 1 : 4;
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
            goto LABEL_26;
          default:
            goto LABEL_19;
        }
      }
      goto LABEL_19;
    }
    if ( a2 == 191 )
LABEL_26:
      v12 = 0;
    else
      v12 = a2 != 215;
  }
LABEL_20:
  if ( v17 != v5 )
    _libc_free((unsigned __int64)v17);
  return v12;
}

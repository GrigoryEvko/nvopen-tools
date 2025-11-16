// Function: sub_2166AE0
// Address: 0x2166ae0
//
__int64 __fastcall sub_2166AE0(__int64 a1, unsigned int a2, __int64 a3, __int64 **a4, unsigned __int64 a5)
{
  _BYTE *v5; // r9
  int v6; // r13d
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r13
  __int64 (*v11)(); // rax
  unsigned int v12; // r12d
  __int64 (*v14)(void); // rax
  char v15; // al
  char v16; // al
  _BYTE *v17; // [rsp+8h] [rbp-88h]
  _BYTE *v18; // [rsp+8h] [rbp-88h]
  _BYTE *v19; // [rsp+8h] [rbp-88h]
  _BYTE *v20; // [rsp+10h] [rbp-80h] BYREF
  __int64 v21; // [rsp+18h] [rbp-78h]
  _BYTE v22[112]; // [rsp+20h] [rbp-70h] BYREF

  v5 = v22;
  v6 = a5;
  v20 = v22;
  v21 = 0x800000000LL;
  if ( a5 > 8 )
  {
    sub_16CD150((__int64)&v20, v22, a5, 8, a5, (int)v22);
    v5 = v22;
  }
  if ( v6 )
  {
    v8 = (unsigned int)v21;
    v9 = (__int64)&a4[(unsigned int)(v6 - 1) + 1];
    do
    {
      v10 = **a4;
      if ( HIDWORD(v21) <= (unsigned int)v8 )
      {
        v17 = v5;
        sub_16CD150((__int64)&v20, v5, 0, 8, a5, (int)v5);
        v8 = (unsigned int)v21;
        v5 = v17;
      }
      ++a4;
      *(_QWORD *)&v20[8 * v8] = v10;
      v8 = (unsigned int)(v21 + 1);
      LODWORD(v21) = v21 + 1;
    }
    while ( a4 != (__int64 **)v9 );
  }
  if ( a2 == 33 )
  {
    v14 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 24) + 152LL);
    if ( v14 != sub_1D5A370 )
    {
      v18 = v5;
      v15 = v14();
      v5 = v18;
      if ( v15 )
        goto LABEL_16;
    }
    goto LABEL_24;
  }
  if ( a2 == 31 )
  {
    v11 = *(__int64 (**)())(**(_QWORD **)(a1 + 24) + 160LL);
    if ( v11 == sub_2165300 )
      goto LABEL_16;
    v19 = v5;
    v16 = v11();
    v5 = v19;
    if ( v16 )
      goto LABEL_16;
LABEL_24:
    v12 = 4;
    goto LABEL_17;
  }
  if ( a2 > 0x1189 )
  {
LABEL_16:
    v12 = 1;
    goto LABEL_17;
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
            goto LABEL_25;
          default:
            goto LABEL_16;
        }
      }
      goto LABEL_16;
    }
    if ( a2 == 191 )
LABEL_25:
      v12 = 0;
    else
      v12 = a2 != 215;
  }
LABEL_17:
  if ( v20 != v5 )
    _libc_free((unsigned __int64)v20);
  return v12;
}

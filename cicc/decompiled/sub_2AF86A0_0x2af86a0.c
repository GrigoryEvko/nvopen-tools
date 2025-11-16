// Function: sub_2AF86A0
// Address: 0x2af86a0
//
void __fastcall sub_2AF86A0(unsigned __int8 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 **v6; // r15
  unsigned __int8 **v7; // r12
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  unsigned __int64 v10; // r8
  unsigned __int8 *v11; // r13
  _BYTE *v12; // r14
  unsigned __int8 v13; // cl
  _QWORD **v14; // r12
  _BYTE *v15; // r13
  _BYTE *v16; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v17; // [rsp+8h] [rbp-B8h]
  _BYTE v18[176]; // [rsp+10h] [rbp-B0h] BYREF

  v6 = &a1[a2];
  v16 = v18;
  v17 = 0x1000000000LL;
  if ( a1 == v6 )
    return;
  v7 = a1;
  v8 = 16;
  v9 = 0;
  while ( 1 )
  {
    v11 = *v7;
    v12 = 0;
    v13 = **v7;
    if ( v13 > 0x1Cu && (v13 == 61 || v13 == 62) )
    {
      v10 = v9 + 1;
      v12 = (_BYTE *)*((_QWORD *)v11 - 4);
      if ( v9 + 1 <= v8 )
        goto LABEL_5;
    }
    else
    {
      v10 = v9 + 1;
      if ( v9 + 1 <= v8 )
        goto LABEL_5;
    }
    sub_C8D5F0((__int64)&v16, v18, v10, 8u, v10, a6);
    v9 = (unsigned int)v17;
LABEL_5:
    *(_QWORD *)&v16[8 * v9] = v11;
    v9 = (unsigned int)(v17 + 1);
    LODWORD(v17) = v17 + 1;
    if ( *v12 == 63 )
      break;
    if ( v6 == ++v7 )
      goto LABEL_15;
LABEL_7:
    v8 = HIDWORD(v17);
  }
  if ( v9 + 1 > (unsigned __int64)HIDWORD(v17) )
  {
    sub_C8D5F0((__int64)&v16, v18, v9 + 1, 8u, v10, a6);
    v9 = (unsigned int)v17;
  }
  ++v7;
  *(_QWORD *)&v16[8 * v9] = v12;
  v9 = (unsigned int)(v17 + 1);
  LODWORD(v17) = v17 + 1;
  if ( v6 != v7 )
    goto LABEL_7;
LABEL_15:
  v14 = (_QWORD **)v16;
  v15 = &v16[8 * v9];
  if ( v16 != v15 )
  {
    do
    {
      while ( (*v14)[2] )
      {
        if ( v15 == (_BYTE *)++v14 )
          goto LABEL_20;
      }
      sub_B43D60(*v14++);
    }
    while ( v15 != (_BYTE *)v14 );
LABEL_20:
    v15 = v16;
  }
  if ( v15 != v18 )
    _libc_free((unsigned __int64)v15);
}

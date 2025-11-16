// Function: sub_3950490
// Address: 0x3950490
//
__int64 __fastcall sub_3950490(__int64 **a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rax
  unsigned __int8 *v6; // rsi
  __int64 v7; // rax
  char v8; // r8
  __int64 v9; // r13
  unsigned __int64 *v10; // rbx
  unsigned __int64 *v11; // r12
  unsigned __int64 v12; // rdi
  unsigned __int64 *v14; // [rsp+8h] [rbp-118h]
  char v15; // [rsp+8h] [rbp-118h]
  unsigned __int8 *v16; // [rsp+18h] [rbp-108h] BYREF
  __int64 v17[5]; // [rsp+20h] [rbp-100h] BYREF
  int v18; // [rsp+48h] [rbp-D8h]
  unsigned __int64 *v19; // [rsp+50h] [rbp-D0h]
  __int64 v20; // [rsp+58h] [rbp-C8h]
  unsigned __int64 *v21; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v22; // [rsp+78h] [rbp-A8h]
  _BYTE v23[160]; // [rsp+80h] [rbp-A0h] BYREF

  v3 = *(_QWORD *)(a2 - 24);
  v21 = (unsigned __int64 *)v23;
  if ( *(_BYTE *)(v3 + 16) )
    v3 = 0;
  v22 = 0x200000000LL;
  sub_1752100(a2, (__int64)&v21);
  v4 = (unsigned int)v22;
  v14 = v21;
  v5 = sub_16498A0(a2);
  v6 = *(unsigned __int8 **)(a2 + 48);
  v20 = v4;
  v17[3] = v5;
  v7 = *(_QWORD *)(a2 + 40);
  v17[0] = 0;
  v17[1] = v7;
  v17[4] = 0;
  v18 = 0;
  v19 = v14;
  v17[2] = a2 + 24;
  v16 = v6;
  if ( v6 )
  {
    sub_1623A60((__int64)&v16, (__int64)v6, 2);
    if ( v17[0] )
      sub_161E7C0((__int64)v17, v17[0]);
    v17[0] = (__int64)v16;
    if ( v16 )
      sub_1623210((__int64)&v16, v16, (__int64)v17);
  }
  v8 = 1;
  if ( ((*(unsigned __int16 *)(a2 + 18) >> 2) & 0x3FFFDFFF) != 0 )
  {
    v8 = 0;
    if ( ((*(unsigned __int16 *)(a2 + 18) >> 2) & 0x3FFFDFFFu) - 66 <= 2 )
      v8 = sub_394FBE0(a2);
  }
  v15 = v8;
  if ( sub_149CB50(**a1, v3, (unsigned int *)&v16)
    && ((_DWORD)v16 == 118 || (_DWORD)v16 == 262 || (_DWORD)v16 == 371 || (_DWORD)v16 == 267 || v15) )
  {
    switch ( (int)v16 )
    {
      case '^':
        v9 = sub_394FFD0((__int64)a1, a2, v17);
        break;
      case '_':
        v9 = sub_3950080((__int64)a1, a2, v17);
        break;
      case '`':
        v9 = sub_3950120((__int64)a1, a2, v17);
        break;
      case 'o':
      case 'q':
        v9 = sub_3950250((__int64)a1, a2, v17, (int)v16);
        break;
      case 'p':
      case 's':
        v9 = sub_3950160(a1, a2, (__int64)v17);
        break;
      default:
        goto LABEL_17;
    }
  }
  else
  {
LABEL_17:
    v9 = 0;
  }
  if ( v17[0] )
    sub_161E7C0((__int64)v17, v17[0]);
  v10 = v21;
  v11 = &v21[7 * (unsigned int)v22];
  if ( v21 != v11 )
  {
    do
    {
      v12 = *(v11 - 3);
      v11 -= 7;
      if ( v12 )
        j_j___libc_free_0(v12);
      if ( (unsigned __int64 *)*v11 != v11 + 2 )
        j_j___libc_free_0(*v11);
    }
    while ( v10 != v11 );
    v11 = v21;
  }
  if ( v11 != (unsigned __int64 *)v23 )
    _libc_free((unsigned __int64)v11);
  return v9;
}

// Function: sub_14A2710
// Address: 0x14a2710
//
__int64 __fastcall sub_14A2710(__int64 *a1, __int64 a2, __int64 a3, __int64 **a4, unsigned __int64 a5)
{
  int v5; // r13d
  __int64 **v6; // rbx
  __int64 v7; // rdi
  __int64 (__fastcall *v8)(__int64, unsigned int, __int64, __int64 **, unsigned __int64); // rax
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r13
  unsigned int v12; // r12d
  _BYTE *v14; // [rsp+10h] [rbp-80h] BYREF
  __int64 v15; // [rsp+18h] [rbp-78h]
  _BYTE v16[112]; // [rsp+20h] [rbp-70h] BYREF

  v5 = a5;
  v6 = a4;
  v7 = *a1;
  v8 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, __int64 **, unsigned __int64))(*(_QWORD *)v7 + 88LL);
  if ( v8 != sub_14A1180 )
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 **))v8)(v7, a2, a3, a4);
  v14 = v16;
  v15 = 0x800000000LL;
  if ( a5 > 8 )
    sub_16CD150(&v14, v16, a5, 8);
  if ( v5 )
  {
    v9 = (unsigned int)v15;
    v10 = (__int64)&v6[(unsigned int)(v5 - 1) + 1];
    do
    {
      v11 = **v6;
      if ( HIDWORD(v15) <= (unsigned int)v9 )
      {
        sub_16CD150(&v14, v16, 0, 8);
        v9 = (unsigned int)v15;
      }
      ++v6;
      *(_QWORD *)&v14[8 * v9] = v11;
      v9 = (unsigned int)(v15 + 1);
      LODWORD(v15) = v15 + 1;
    }
    while ( (__int64 **)v10 != v6 );
  }
  if ( (unsigned int)a2 <= 0x1189 )
  {
    if ( (unsigned int)a2 > 0x1182 )
    {
      v12 = ((1LL << ((unsigned __int8)a2 + 125)) & 0x49) == 0 ? 1 : 4;
      goto LABEL_15;
    }
    if ( (unsigned int)a2 > 0x95 )
    {
      if ( (_DWORD)a2 == 191 )
LABEL_21:
        v12 = 0;
      else
        v12 = a2 != 215;
      goto LABEL_15;
    }
    if ( (unsigned int)a2 > 2 )
    {
      switch ( (int)a2 )
      {
        case 3:
        case 4:
        case 14:
        case 15:
        case 18:
        case 19:
        case 20:
        case 23:
        case 27:
        case 28:
        case 29:
        case 36:
        case 37:
        case 38:
        case 76:
        case 77:
        case 113:
        case 114:
        case 116:
        case 117:
        case 144:
        case 149:
          goto LABEL_21;
        default:
          break;
      }
    }
  }
  v12 = 1;
LABEL_15:
  if ( v14 != v16 )
    _libc_free((unsigned __int64)v14);
  return v12;
}

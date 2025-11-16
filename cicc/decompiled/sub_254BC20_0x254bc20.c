// Function: sub_254BC20
// Address: 0x254bc20
//
__int64 __fastcall sub_254BC20(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int64 a4, _BYTE *a5, _BYTE *a6)
{
  __int64 v7; // rsi
  unsigned int v11; // r14d
  bool v12; // al
  unsigned int v13; // r14d
  bool v14; // al
  unsigned int v15; // r14d
  bool v16; // al
  unsigned int v17; // r14d
  bool v18; // al
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-28h]

  v7 = a3;
  switch ( *a2 )
  {
    case '*':
      sub_9865C0((__int64)&v21, a3);
      sub_C45EE0((__int64)&v21, (__int64 *)a4);
      *(_DWORD *)(a1 + 8) = v22;
      *(_QWORD *)a1 = v21;
      return a1;
    case ',':
      sub_9865C0((__int64)&v21, a3);
      sub_C46B40((__int64)&v21, (__int64 *)a4);
      *(_DWORD *)(a1 + 8) = v22;
      *(_QWORD *)a1 = v21;
      return a1;
    case '.':
      sub_C472A0(a1, a3, (__int64 *)a4);
      return a1;
    case '0':
      v11 = *(_DWORD *)(a4 + 8);
      if ( v11 <= 0x40 )
      {
        v12 = *(_QWORD *)a4 == 0;
      }
      else
      {
        v7 = a3;
        v12 = v11 == (unsigned int)sub_C444A0(a4);
      }
      if ( v12 )
        goto LABEL_40;
      sub_C4A1D0(a1, v7, a4);
      return a1;
    case '1':
      v13 = *(_DWORD *)(a4 + 8);
      if ( v13 <= 0x40 )
      {
        v14 = *(_QWORD *)a4 == 0;
      }
      else
      {
        v7 = a3;
        v14 = v13 == (unsigned int)sub_C444A0(a4);
      }
      if ( v14 )
        goto LABEL_40;
      sub_C4A3E0(a1, v7, a4);
      return a1;
    case '3':
      v15 = *(_DWORD *)(a4 + 8);
      if ( v15 <= 0x40 )
      {
        v16 = *(_QWORD *)a4 == 0;
      }
      else
      {
        v7 = a3;
        v16 = v15 == (unsigned int)sub_C444A0(a4);
      }
      if ( v16 )
        goto LABEL_40;
      sub_C4B490(a1, v7, a4);
      return a1;
    case '4':
      v17 = *(_DWORD *)(a4 + 8);
      if ( v17 <= 0x40 )
      {
        v18 = *(_QWORD *)a4 == 0;
      }
      else
      {
        v7 = a3;
        v18 = v17 == (unsigned int)sub_C444A0(a4);
      }
      if ( v18 )
      {
LABEL_40:
        *a5 = 1;
        sub_9865C0(a1, v7);
        return a1;
      }
      else
      {
        sub_C4B8A0(a1, v7, a4);
        return a1;
      }
    case '6':
      sub_9865C0(a1, a3);
      sub_C47AC0(a1, a4);
      return a1;
    case '7':
      sub_9865C0(a1, a3);
      sub_C48380(a1, a4);
      return a1;
    case '8':
      sub_9865C0(a1, a3);
      sub_C44D10(a1, a4);
      return a1;
    case '9':
      sub_9865C0((__int64)&v21, a3);
      v19 = v22;
      if ( v22 > 0x40 )
      {
        sub_C43B90(&v21, (__int64 *)a4);
        v19 = v22;
        v20 = v21;
      }
      else
      {
        v20 = *(_QWORD *)a4 & v21;
      }
      goto LABEL_32;
    case ':':
      sub_9865C0((__int64)&v21, a3);
      v19 = v22;
      if ( v22 > 0x40 )
      {
        sub_C43BD0(&v21, (__int64 *)a4);
        v19 = v22;
        v20 = v21;
      }
      else
      {
        v20 = *(_QWORD *)a4 | v21;
      }
      goto LABEL_32;
    case ';':
      sub_9865C0((__int64)&v21, a3);
      v19 = v22;
      if ( v22 > 0x40 )
      {
        sub_C43C10(&v21, (__int64 *)a4);
        v19 = v22;
        v20 = v21;
      }
      else
      {
        v20 = *(_QWORD *)a4 ^ v21;
      }
LABEL_32:
      *(_DWORD *)(a1 + 8) = v19;
      *(_QWORD *)a1 = v20;
      return a1;
    default:
      *a6 = 1;
      sub_9865C0(a1, a3);
      return a1;
  }
}

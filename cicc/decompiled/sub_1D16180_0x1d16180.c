// Function: sub_1D16180
// Address: 0x1d16180
//
__int64 __fastcall sub_1D16180(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v3; // rsi
  unsigned __int64 v4; // rcx
  __int64 v5; // r15
  __int64 *v7; // rsi
  __int16 *v8; // r12
  __int64 v9; // rax
  __int16 *v10; // rsi
  unsigned int v11; // r14d
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rbx
  _QWORD v16[2]; // [rsp+0h] [rbp-50h] BYREF
  char v17[8]; // [rsp+10h] [rbp-40h] BYREF
  __int16 *v18; // [rsp+18h] [rbp-38h] BYREF
  __int64 v19; // [rsp+20h] [rbp-30h]

  v7 = (__int64 *)(a3 + 8);
  v8 = (__int16 *)sub_16982C0();
  if ( (__int16 *)*v7 == v8 )
    sub_169C6E0(&v18, (__int64)v7);
  else
    sub_16986C0(&v18, v7);
  v16[0] = a1;
  v16[1] = a2;
  if ( !(_BYTE)a1 )
  {
    v9 = sub_1F58D20(v16);
    if ( !(_BYTE)v9 )
    {
LABEL_23:
      v3 = *(unsigned __int64 **)(v9 + 8);
      v4 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
      *v3 = v4 | *v3 & 7;
      *(_QWORD *)(v4 + 8) = v3;
      *(_QWORD *)v9 &= 7uLL;
      *(_QWORD *)(v9 + 8) = 0;
      nullsub_686(0, v5, 0);
      BUG();
    }
    v9 = sub_1F596B0(v16);
    LOBYTE(a1) = v9;
LABEL_5:
    switch ( (char)a1 )
    {
      case 8:
        goto LABEL_17;
      case 9:
        goto LABEL_18;
      case 10:
        goto LABEL_16;
      case 11:
        v10 = (__int16 *)sub_16982A0();
        goto LABEL_11;
      case 12:
        v10 = (__int16 *)sub_1698290();
        goto LABEL_11;
      case 13:
        v10 = v8;
        goto LABEL_11;
      default:
        goto LABEL_23;
    }
  }
  v9 = (unsigned int)(a1 - 14);
  if ( (unsigned __int8)(a1 - 14) > 0x5Fu )
    goto LABEL_5;
  switch ( (char)a1 )
  {
    case 'V':
    case 'W':
    case 'X':
    case 'b':
    case 'c':
    case 'd':
LABEL_17:
      v10 = (__int16 *)sub_1698260();
      goto LABEL_11;
    case 'Y':
    case 'Z':
    case '[':
    case '\\':
    case ']':
    case 'e':
    case 'f':
    case 'g':
    case 'h':
    case 'i':
LABEL_18:
      v10 = (__int16 *)sub_1698270();
      goto LABEL_11;
    case '^':
    case '_':
    case '`':
    case 'a':
    case 'j':
    case 'k':
    case 'l':
    case 'm':
LABEL_16:
      v10 = (__int16 *)sub_1698280();
LABEL_11:
      sub_16A3360((__int64)v17, v10, 0, (bool *)v16);
      v11 = LOBYTE(v16[0]) ^ 1;
      if ( v18 != v8 )
      {
        sub_1698460((__int64)&v18);
        return v11;
      }
      v13 = v19;
      if ( !v19 )
        return v11;
      v14 = 32LL * *(_QWORD *)(v19 - 8);
      v15 = v19 + v14;
      if ( v19 != v19 + v14 )
      {
        do
        {
          v15 -= 32;
          sub_127D120((_QWORD *)(v15 + 8));
        }
        while ( v13 != v15 );
      }
      j_j_j___libc_free_0_0(v13 - 8);
      return v11;
    default:
      goto LABEL_23;
  }
}

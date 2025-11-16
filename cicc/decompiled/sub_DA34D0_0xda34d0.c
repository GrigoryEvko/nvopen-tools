// Function: sub_DA34D0
// Address: 0xda34d0
//
__int64 __fastcall sub_DA34D0(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned int v7; // r12d
  unsigned int v8; // eax
  __int64 v10; // rax
  int v11; // eax
  bool v12; // zf
  bool v13; // sf
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-48h] BYREF
  __int64 v18; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-38h]
  unsigned __int64 v20; // [rsp+20h] [rbp-30h] BYREF
  int v21; // [rsp+28h] [rbp-28h]

  v17 = a1;
  v19 = 1;
  v18 = 0;
  v21 = 1;
  v20 = 0;
  v5 = a4;
  switch ( a2 )
  {
    case '"':
      v16 = a3;
      a3 = a4;
      v5 = v16;
      goto LABEL_19;
    case '#':
      v10 = a3;
      a3 = a4;
      v5 = v10;
      goto LABEL_13;
    case '$':
LABEL_19:
      v7 = sub_DA3330(&v17, a3, v5, (__int64)&v18, (__int64)&v20, 2);
      if ( (_BYTE)v7 )
        v7 = (unsigned int)sub_C49970((__int64)&v18, &v20) >> 31;
      goto LABEL_4;
    case '%':
LABEL_13:
      v7 = sub_DA3330(&v17, a3, v5, (__int64)&v18, (__int64)&v20, 2);
      if ( !(_BYTE)v7 )
        goto LABEL_4;
      v11 = sub_C49970((__int64)&v18, &v20);
      v12 = v11 == 0;
      v13 = v11 < 0;
      v8 = v21;
      LOBYTE(v7) = v13 || v12;
      goto LABEL_5;
    case '&':
      v6 = a3;
      a3 = a4;
      v5 = v6;
      goto LABEL_3;
    case '\'':
      v14 = a3;
      a3 = a4;
      v5 = v14;
      goto LABEL_16;
    case '(':
LABEL_3:
      v7 = sub_DA3330(&v17, a3, v5, (__int64)&v18, (__int64)&v20, 4);
      if ( !(_BYTE)v7 )
        goto LABEL_4;
      v7 = (unsigned int)sub_C4C880((__int64)&v18, (__int64)&v20) >> 31;
      v8 = v21;
      goto LABEL_5;
    case ')':
LABEL_16:
      v7 = sub_DA3330(&v17, a3, v5, (__int64)&v18, (__int64)&v20, 4);
      if ( (_BYTE)v7 )
      {
        v15 = sub_C4C880((__int64)&v18, (__int64)&v20);
        v12 = v15 == 0;
        v13 = v15 < 0;
        v8 = v21;
        LOBYTE(v7) = v13 || v12;
      }
      else
      {
LABEL_4:
        v8 = v21;
      }
LABEL_5:
      if ( v8 > 0x40 && v20 )
        j_j___libc_free_0_0(v20);
      break;
    default:
      v7 = 0;
      break;
  }
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return v7;
}

// Function: sub_B6DC00
// Address: 0xb6dc00
//
__int64 __fastcall sub_B6DC00(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  int v6; // r9d
  __int64 v7; // rax
  int v8; // r9d
  unsigned __int64 v9; // r8
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  unsigned int v14; // eax
  _BYTE *v15; // rsi
  __int64 v16; // r12
  __int64 v18; // [rsp+8h] [rbp-118h]
  __int64 v19; // [rsp+18h] [rbp-108h]
  _BYTE *v20; // [rsp+20h] [rbp-100h] BYREF
  unsigned __int64 v21; // [rsp+28h] [rbp-F8h]
  _BYTE *v22; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v23; // [rsp+38h] [rbp-E8h]
  _BYTE v24[64]; // [rsp+40h] [rbp-E0h] BYREF
  _BYTE *v25; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v26; // [rsp+88h] [rbp-98h]
  _BYTE v27[144]; // [rsp+90h] [rbp-90h] BYREF

  v26 = 0x800000000LL;
  v25 = v27;
  sub_B6DAB0(a2, (__int64)&v25);
  v20 = v25;
  v21 = (unsigned int)v26;
  v7 = sub_B5E480((__int64)&v20, a3, a1, v4, v5, v6);
  v9 = v21;
  v10 = 0x800000000LL;
  v22 = v24;
  v19 = v7;
  v23 = 0x800000000LL;
  if ( !v21 )
  {
    v15 = v24;
    goto LABEL_7;
  }
  do
  {
    v11 = sub_B5E480((__int64)&v20, a3, a1, v10, v9, v8);
    v12 = (unsigned int)v23;
    v9 = (unsigned int)v23 + 1LL;
    if ( v9 > HIDWORD(v23) )
    {
      v18 = v11;
      sub_C8D5F0(&v22, v24, (unsigned int)v23 + 1LL, 8);
      v12 = (unsigned int)v23;
      v11 = v18;
    }
    v10 = (__int64)v22;
    *(_QWORD *)&v22[8 * v12] = v11;
    v13 = (unsigned int)v23;
    v14 = v23 + 1;
    LODWORD(v23) = v23 + 1;
  }
  while ( v21 );
  v15 = v22;
  v9 = v14;
  if ( !v14 )
  {
    v9 = 0;
    goto LABEL_7;
  }
  if ( *(_BYTE *)(*(_QWORD *)&v22[8 * v14 - 8] + 8LL) != 7 )
  {
LABEL_7:
    v16 = sub_BCF480(v19, v15, v9, 0);
    goto LABEL_8;
  }
  LODWORD(v23) = v13;
  v16 = sub_BCF480(v19, v22, v13, 1);
LABEL_8:
  if ( v22 != v24 )
    _libc_free(v22, v15);
  if ( v25 != v27 )
    _libc_free(v25, v15);
  return v16;
}

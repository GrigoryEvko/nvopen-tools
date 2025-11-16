// Function: sub_38D3660
// Address: 0x38d3660
//
__int64 __fastcall sub_38D3660(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r12
  char *v3; // r13
  size_t v4; // r14
  _QWORD *v5; // rax
  unsigned __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // r12
  unsigned __int64 v10[2]; // [rsp+10h] [rbp-90h] BYREF
  char v11; // [rsp+20h] [rbp-80h]
  char v12; // [rsp+21h] [rbp-7Fh]
  unsigned __int64 *v13; // [rsp+30h] [rbp-70h] BYREF
  __int16 v14; // [rsp+40h] [rbp-60h]
  char v15; // [rsp+44h] [rbp-5Ch] BYREF
  _BYTE v16[11]; // [rsp+45h] [rbp-5Bh] BYREF
  unsigned __int64 v17[2]; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v18[8]; // [rsp+60h] [rbp-40h] BYREF

  if ( *(_DWORD *)(a1 + 748) != 2 )
    sub_16BD130("Cannot get DWARF types section for this object file format: not implemented.", 1u);
  v2 = *(_QWORD *)(a1 + 688);
  if ( !a2 )
  {
    v15 = 48;
    v3 = &v15;
    v17[0] = (unsigned __int64)v18;
LABEL_4:
    v4 = 1;
    LOBYTE(v18[0]) = *v3;
    v5 = v18;
    goto LABEL_10;
  }
  v3 = v16;
  do
  {
    *--v3 = a2 % 0xA + 48;
    v6 = a2;
    a2 /= 0xAu;
  }
  while ( v6 > 9 );
  v4 = v16 - v3;
  v17[0] = (unsigned __int64)v18;
  v10[0] = v16 - v3;
  if ( (unsigned __int64)(v16 - v3) > 0xF )
  {
    v17[0] = sub_22409D0((__int64)v17, v10, 0);
    v7 = (_QWORD *)v17[0];
    v18[0] = v10[0];
LABEL_9:
    memcpy(v7, v3, v4);
    v4 = v10[0];
    v5 = (_QWORD *)v17[0];
    goto LABEL_10;
  }
  if ( v4 == 1 )
    goto LABEL_4;
  if ( v4 )
  {
    v7 = v18;
    goto LABEL_9;
  }
  v5 = v18;
LABEL_10:
  v17[1] = v4;
  *((_BYTE *)v5 + v4) = 0;
  v14 = 260;
  v13 = v17;
  v12 = 1;
  v10[0] = (unsigned __int64)".debug_types";
  v11 = 3;
  v8 = sub_38C3B80(v2, (__int64)v10, 1, 512, 0, (__int64)&v13, -1, 0);
  if ( (_QWORD *)v17[0] != v18 )
    j_j___libc_free_0(v17[0]);
  return v8;
}

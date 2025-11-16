// Function: sub_23DC880
// Address: 0x23dc880
//
__int64 __fastcall sub_23DC880(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned int v6; // eax
  unsigned int v7; // edx
  _QWORD *v8; // rdx
  unsigned __int64 v9; // rdi
  __int64 v10; // rcx
  _QWORD v11[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v12; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-28h]
  unsigned __int64 v14; // [rsp+20h] [rbp-20h]
  unsigned int v15; // [rsp+28h] [rbp-18h]

  v11[1] = a4;
  v4 = (unsigned __int8)a4;
  v11[0] = a3;
  if ( (_BYTE)a4 )
    return 0;
  sub_D62AB0((__int64)&v12, a1, a2);
  v6 = v13;
  v7 = v13;
  if ( v13 <= 1 )
    goto LABEL_13;
  if ( v15 <= 1 )
    goto LABEL_20;
  v8 = v12;
  if ( v13 > 0x40 )
    v8 = (_QWORD *)*v12;
  v9 = v14;
  if ( v15 > 0x40 )
  {
    v10 = *(_QWORD *)v14;
    LOBYTE(v4) = *(_QWORD *)v14 >= 0LL && *(_QWORD *)v14 <= (unsigned __int64)v8;
    if ( !(_BYTE)v4 )
    {
LABEL_10:
      j_j___libc_free_0_0(v9);
LABEL_15:
      v6 = v13;
      goto LABEL_16;
    }
    goto LABEL_12;
  }
  v10 = (__int64)(v14 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
  v4 = v10 >= 0;
  LOBYTE(v4) = (v10 <= (unsigned __int64)v8) & v4;
  if ( (_BYTE)v4 )
  {
LABEL_12:
    v4 = (_DWORD)v8 - v10;
    LOBYTE(v4) = (unsigned __int64)v8 - v10 >= (unsigned __int64)sub_CA1930(v11) >> 3;
LABEL_13:
    if ( v15 > 0x40 )
    {
      v9 = v14;
      if ( !v14 )
        goto LABEL_15;
      goto LABEL_10;
    }
    v7 = v13;
LABEL_20:
    v6 = v7;
  }
LABEL_16:
  if ( v6 <= 0x40 || !v12 )
    return v4;
  j_j___libc_free_0_0((unsigned __int64)v12);
  return v4;
}

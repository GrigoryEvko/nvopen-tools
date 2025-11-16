// Function: sub_C473B0
// Address: 0xc473b0
//
__int64 __fastcall sub_C473B0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // eax
  unsigned int v4; // r15d
  __int64 v5; // r12
  unsigned __int64 v6; // r12
  __int64 v8; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-58h]
  __int64 v10; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v11; // [rsp+28h] [rbp-48h]
  __int64 v12; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+38h] [rbp-38h]

  v2 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v2;
  if ( v2 > 0x40 )
    sub_C43780(a1, (const void **)a2);
  else
    *(_QWORD *)a1 = *(_QWORD *)a2;
  sub_C472A0((__int64)&v12, a2, (__int64 *)a1);
  while ( 1 )
  {
    v4 = v13;
    v5 = v12;
    v13 = 0;
    v8 = v12;
    v9 = v4;
    if ( v4 > 0x40 )
      break;
    if ( v12 == 1 )
      return a1;
    v11 = v4;
    v6 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & ~v12;
    if ( !v4 )
      v6 = 0;
    v10 = v6;
LABEL_6:
    sub_C46250((__int64)&v10);
    sub_C46A40((__int64)&v10, 2);
    v3 = v11;
    v11 = 0;
    v13 = v3;
    v12 = v10;
    sub_C47360(a1, &v12);
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    if ( v11 > 0x40 )
    {
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
    sub_C472A0((__int64)&v12, a2, (__int64 *)a1);
  }
  if ( (unsigned int)sub_C444A0((__int64)&v8) != v4 - 1 )
  {
    v10 = v5;
    v11 = v4;
    sub_C43D10((__int64)&v10);
    goto LABEL_6;
  }
  if ( v5 )
    j_j___libc_free_0_0(v5);
  return a1;
}

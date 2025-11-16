// Function: sub_2F9A750
// Address: 0x2f9a750
//
__int64 __fastcall sub_2F9A750(__int64 a1)
{
  unsigned int v1; // r13d
  unsigned int v2; // eax
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rdx
  unsigned int v5; // ebx
  _QWORD *v6; // r12
  _QWORD *v8; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v9; // [rsp+8h] [rbp-38h]
  unsigned __int64 v10; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-28h]

  v2 = *(_DWORD *)(a1 + 32);
  v9 = v2;
  if ( v2 > 0x40 )
  {
    sub_C43780((__int64)&v8, (const void **)(a1 + 24));
    v2 = v9;
    if ( v9 > 0x40 )
    {
      sub_C43D10((__int64)&v8);
      goto LABEL_6;
    }
    v3 = (unsigned __int64)v8;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 24);
  }
  v4 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & ~v3;
  if ( !v2 )
    v4 = 0;
  v8 = (_QWORD *)v4;
LABEL_6:
  sub_C46250((__int64)&v8);
  v5 = v9;
  v6 = v8;
  v9 = 0;
  v11 = v5;
  v10 = (unsigned __int64)v8;
  if ( v5 <= 0x40 )
  {
    LOBYTE(v1) = v8 == (_QWORD *)1;
    return v1;
  }
  if ( v5 - (unsigned int)sub_C444A0((__int64)&v10) <= 0x40 && *v6 == 1 )
  {
    v1 = 1;
  }
  else
  {
    v1 = 0;
    if ( !v6 )
      return v1;
  }
  j_j___libc_free_0_0((unsigned __int64)v6);
  if ( v9 <= 0x40 || !v8 )
    return v1;
  j_j___libc_free_0_0((unsigned __int64)v8);
  return v1;
}

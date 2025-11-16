// Function: sub_22CE0E0
// Address: 0x22ce0e0
//
unsigned __int64 __fastcall sub_22CE0E0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  int v7; // eax
  __int64 v8; // r12
  __int64 *v10; // rax
  _BYTE v11[8]; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int64 v12; // [rsp+8h] [rbp-48h] BYREF
  unsigned int v13; // [rsp+10h] [rbp-40h]
  unsigned __int64 v14; // [rsp+18h] [rbp-38h]
  unsigned int v15; // [rsp+20h] [rbp-30h]

  if ( *sub_BD3990((unsigned __int8 *)a2, a2) == 60 )
    return 0;
  v4 = *(_QWORD *)(a3 + 40);
  v5 = sub_AA4B30(v4);
  v6 = sub_22C1480(a1, v5);
  sub_22CDEF0((__int64)v11, v6, a2, v4, a3);
  v7 = v11[0];
  if ( v11[0] == 2 )
    return v12;
  v8 = 0;
  if ( (unsigned __int8)(v11[0] - 4) <= 1u )
  {
    v10 = sub_9876C0((__int64 *)&v12);
    v8 = (__int64)v10;
    if ( v10 )
      v8 = sub_AD8D80(*(_QWORD *)(a2 + 8), (__int64)v10);
    v7 = v11[0];
  }
  if ( (unsigned int)(v7 - 4) <= 1 )
  {
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    if ( v13 > 0x40 )
    {
      if ( v12 )
        j_j___libc_free_0_0(v12);
    }
  }
  return v8;
}

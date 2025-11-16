// Function: sub_3260DB0
// Address: 0x3260db0
//
__int64 __fastcall sub_3260DB0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r13
  __int64 v4; // rsi
  unsigned int v5; // ebx
  unsigned __int64 v6; // r12
  const void *v7; // r12
  __int64 v8; // rdi
  bool v9; // cc
  int v10; // eax
  const void *v12; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-38h]
  const void *v14; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-28h]

  v3 = *a3;
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v5 = *(_DWORD *)(v4 + 32);
  v13 = v5;
  if ( v5 > 0x40 )
  {
    sub_C43780((__int64)&v12, (const void **)(v4 + 24));
    v5 = v13;
    if ( v13 > 0x40 )
    {
      sub_C43D10((__int64)&v12);
      v5 = v13;
      v7 = v12;
      goto LABEL_6;
    }
    v6 = (unsigned __int64)v12;
  }
  else
  {
    v6 = *(_QWORD *)(v4 + 24);
  }
  v7 = (const void *)((0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & ~v6);
  if ( !v5 )
    v7 = 0;
  v12 = v7;
LABEL_6:
  v8 = *(_QWORD *)(v3 + 96);
  v15 = v5;
  v14 = v7;
  v9 = *(_DWORD *)(v8 + 32) <= 0x40u;
  v13 = 0;
  if ( v9 )
  {
    LOBYTE(v3) = *(_QWORD *)(v8 + 24) == (_QWORD)v7;
  }
  else
  {
    LOBYTE(v10) = sub_C43C50(v8 + 24, &v14);
    LODWORD(v3) = v10;
  }
  if ( v5 > 0x40 )
  {
    if ( v7 )
    {
      j_j___libc_free_0_0((unsigned __int64)v7);
      if ( v13 > 0x40 )
      {
        if ( v12 )
          j_j___libc_free_0_0((unsigned __int64)v12);
      }
    }
  }
  return (unsigned int)v3;
}

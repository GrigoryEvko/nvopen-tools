// Function: sub_29CFC90
// Address: 0x29cfc90
//
__int64 __fastcall sub_29CFC90(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v4; // r13
  unsigned int v5; // eax
  unsigned int v6; // edx
  __int64 result; // rax
  __int64 v8; // [rsp+8h] [rbp-48h]
  char *v9; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-38h]
  char *v11; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-28h]

  v10 = sub_AE43F0(*(_QWORD *)(a1 + 624), *(_QWORD *)(a2 + 8));
  if ( v10 > 0x40 )
    sub_C43690((__int64)&v9, 0, 0);
  else
    v9 = 0;
  v4 = sub_BD45C0((unsigned __int8 *)a2, *(_QWORD *)(a1 + 624), (__int64)&v9, 1, 0, 0, 0, 0);
  v5 = sub_AE43F0(*(_QWORD *)(a1 + 624), *((_QWORD *)v4 + 1));
  sub_C44B10((__int64)&v11, &v9, v5);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0((unsigned __int64)v9);
  v6 = v12;
  v9 = v11;
  result = 0;
  v10 = v12;
  if ( *v4 == 3 )
  {
    result = sub_29CFB10(a1, (__int64)v4, a3, (__int64)&v9);
    v6 = v10;
  }
  if ( v6 > 0x40 )
  {
    if ( v9 )
    {
      v8 = result;
      j_j___libc_free_0_0((unsigned __int64)v9);
      return v8;
    }
  }
  return result;
}

// Function: sub_2487D40
// Address: 0x2487d40
//
__int64 __fastcall sub_2487D40(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  volatile signed __int32 *v6; // rdi
  volatile signed __int32 *v7; // [rsp+8h] [rbp-18h] BYREF

  *a1 = (__int64)(a1 + 2);
  sub_2484EB0(a1, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  result = *a3;
  a1[4] = *a3;
  if ( result )
    _InterlockedAdd((volatile signed __int32 *)(result + 8), 1u);
  if ( !*a3 )
  {
    sub_CA41E0(&v7);
    result = (__int64)v7;
    v6 = (volatile signed __int32 *)a1[4];
    a1[4] = (__int64)v7;
    v7 = v6;
    if ( v6 )
    {
      if ( !_InterlockedSub(v6 + 2, 1u) )
        return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v6 + 8LL))(v6);
    }
  }
  return result;
}

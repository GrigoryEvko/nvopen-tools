// Function: sub_1ED0E70
// Address: 0x1ed0e70
//
__int64 __fastcall sub_1ED0E70(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned int v3; // ecx
  __int64 v4; // rax
  _QWORD *v5; // r12
  __int64 result; // rax
  volatile signed __int32 *v7; // rdi
  volatile signed __int32 *v8; // rbx
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  volatile signed __int32 *v10; // [rsp+8h] [rbp-28h]
  unsigned int v11; // [rsp+10h] [rbp-20h] BYREF
  __int64 v12; // [rsp+18h] [rbp-18h]

  v3 = *(_DWORD *)a3;
  v4 = *(_QWORD *)(a3 + 8);
  *(_DWORD *)a3 = 0;
  *(_QWORD *)(a3 + 8) = 0;
  v11 = v3;
  v12 = v4;
  sub_1ED0750(&v9, a1 + 88, &v11);
  if ( v12 )
    j_j___libc_free_0_0(v12);
  v5 = (_QWORD *)(*(_QWORD *)(a1 + 160) + 88LL * a2);
  result = v9;
  v7 = (volatile signed __int32 *)v5[1];
  *v5 = v9;
  v8 = v10;
  if ( v10 != v7 )
  {
    if ( v10 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(v10 + 2, 1u);
      else
        ++*((_DWORD *)v10 + 2);
      v7 = (volatile signed __int32 *)v5[1];
    }
    if ( v7 )
      result = sub_A191D0(v7);
    v5[1] = v8;
    v7 = v10;
  }
  if ( v7 )
    return sub_A191D0(v7);
  return result;
}

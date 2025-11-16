// Function: sub_9C14F0
// Address: 0x9c14f0
//
__int64 __fastcall sub_9C14F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v4; // rdi
  _QWORD *v5; // r9
  __int64 v6; // rax
  __int64 v8; // [rsp+0h] [rbp-70h] BYREF
  __int64 v9; // [rsp+8h] [rbp-68h]
  __int64 v10; // [rsp+10h] [rbp-60h]
  __int64 v11; // [rsp+18h] [rbp-58h]
  _QWORD *v12; // [rsp+20h] [rbp-50h]
  __int64 v13; // [rsp+28h] [rbp-48h]
  _BYTE v14[64]; // [rsp+30h] [rbp-40h] BYREF

  v2 = a2;
  if ( !a1 )
    return v2;
  if ( a1 != a2 && a2 )
  {
    v13 = 0x400000000LL;
    v8 = 0;
    v9 = 0;
    v10 = 0;
    v11 = 0;
    v12 = v14;
    sub_9C1440((__int64)&v8, a1);
    v2 = 0;
    sub_9C1440((__int64)&v8, a2);
    v4 = v12;
    if ( (_DWORD)v13 )
    {
      if ( (unsigned int)v13 == 1 )
      {
        v2 = *v12;
      }
      else
      {
        v5 = (_QWORD *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
          v5 = (_QWORD *)*v5;
        a2 = (__int64)v12;
        v6 = sub_B9C770(v5, v12, (unsigned int)v13, 0, 1);
        v4 = v12;
        v2 = v6;
      }
    }
    if ( v4 != v14 )
      _libc_free(v4, a2);
    sub_C7D6A0(v9, 8LL * (unsigned int)v11, 8);
    return v2;
  }
  return a1;
}

// Function: sub_14799E0
// Address: 0x14799e0
//
__int64 __fastcall sub_14799E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned int v6; // ecx
  bool v7; // zf
  __int64 v8; // r12
  char v10; // [rsp+Ch] [rbp-54h]
  __int64 *v11; // [rsp+10h] [rbp-50h] BYREF
  __int64 v12; // [rsp+18h] [rbp-48h]
  _QWORD v13[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = a5;
  v7 = *(_WORD *)(a3 + 24) == 7;
  v11 = v13;
  v13[0] = a2;
  v12 = 0x400000001LL;
  if ( v7 && *(_QWORD *)(a3 + 48) == a4 )
  {
    v10 = a5;
    sub_145C5B0((__int64)&v11, *(_BYTE **)(a3 + 32), (_BYTE *)(*(_QWORD *)(a3 + 32) + 8LL * *(_QWORD *)(a3 + 40)));
    v6 = v10 & 1;
  }
  else
  {
    v13[1] = a3;
    LODWORD(v12) = 2;
  }
  v8 = sub_14785F0(a1, &v11, a4, v6);
  if ( v11 != v13 )
    _libc_free((unsigned __int64)v11);
  return v8;
}

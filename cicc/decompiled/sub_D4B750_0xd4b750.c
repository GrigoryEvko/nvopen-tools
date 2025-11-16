// Function: sub_D4B750
// Address: 0xd4b750
//
__int64 __fastcall sub_D4B750(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  bool v3; // al
  __int64 v4; // rax
  unsigned int v5; // r12d
  _QWORD v7[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v8; // [rsp+10h] [rbp-60h]
  int v9; // [rsp+18h] [rbp-58h]
  __int64 v10; // [rsp+20h] [rbp-50h]
  _BYTE *v11; // [rsp+28h] [rbp-48h]
  _BYTE *v12; // [rsp+30h] [rbp-40h]
  __int64 v13; // [rsp+38h] [rbp-38h]
  _BYTE v14[48]; // [rsp+40h] [rbp-30h] BYREF

  v7[0] = 6;
  v7[1] = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = v14;
  v13 = 0x200000000LL;
  if ( !(unsigned __int8)sub_D4B6F0(a1, a2, (__int64)v7)
    || !v8
    || *(_BYTE *)v8 != 17
    || ((v2 = *(_DWORD *)(v8 + 32), v2 <= 0x40)
      ? (v3 = *(_QWORD *)(v8 + 24) == 0)
      : (v3 = v2 == (unsigned int)sub_C444A0(v8 + 24)),
        !v3
     || !v11
     || *v11 != 42
     || (v4 = sub_1023590(v7)) == 0
     || ((v5 = *(_DWORD *)(v4 + 32), v5 <= 0x40)
       ? (LOBYTE(v5) = *(_QWORD *)(v4 + 24) == 1)
       : (--v5, LOBYTE(v5) = v5 == (unsigned int)sub_C444A0(v4 + 24)),
         !(_BYTE)v5)) )
  {
    v5 = 0;
  }
  if ( v12 != v14 )
    _libc_free(v12, a2);
  if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
    sub_BD60C0(v7);
  return v5;
}

// Function: sub_2AAE050
// Address: 0x2aae050
//
__int64 __fastcall sub_2AAE050(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // dl
  char v4; // bl
  __int64 v5; // rax
  unsigned __int8 v6; // dl
  unsigned int v7; // r8d
  unsigned int v8; // esi
  __int64 v10; // [rsp+0h] [rbp-40h]
  char v11; // [rsp+8h] [rbp-38h]

  v2 = sub_9208B0(a2, a1);
  v11 = v3;
  v10 = v2;
  v4 = sub_AE5020(a2, a1);
  v5 = sub_9208B0(a2, a1);
  v7 = 1;
  v8 = v6;
  if ( v10 == 8 * (((1LL << v4) + ((unsigned __int64)(v5 + 7) >> 3) - 1) >> v4 << v4) )
  {
    LOBYTE(v8) = v11 ^ v6;
    return v8;
  }
  return v7;
}

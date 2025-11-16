// Function: sub_2919770
// Address: 0x2919770
//
bool __fastcall sub_2919770(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  char v3; // dl
  char v4; // r14
  __int64 v5; // rax
  char v6; // dl
  char v7; // di
  __int64 v8; // rdx
  bool result; // al

  v2 = sub_9208B0(a1, a2) + 7;
  v4 = v3;
  v5 = sub_9208B0(a1, a2);
  v7 = v6;
  v8 = v5;
  result = 0;
  if ( v8 == (v2 & 0xFFFFFFFFFFFFFFF8LL) )
    return v7 == v4;
  return result;
}

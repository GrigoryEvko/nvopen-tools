// Function: sub_D47630
// Address: 0xd47630
//
__int64 __fastcall sub_D47630(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdi
  __int64 v3; // rax
  unsigned int v4; // r8d
  char v5; // dl
  __int64 v7; // [rsp+0h] [rbp-10h] BYREF
  __int64 *v8; // [rsp+8h] [rbp-8h] BYREF

  v1 = *(_QWORD *)(a1 + 40);
  v7 = a1;
  v2 = *(_QWORD *)(a1 + 32);
  v8 = &v7;
  v3 = sub_D46690(v2, v1, &v8, 0);
  v4 = 0;
  if ( !v5 )
    LOBYTE(v4) = v3 == 0;
  return v4;
}

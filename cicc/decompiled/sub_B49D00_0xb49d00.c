// Function: sub_B49D00
// Address: 0xb49d00
//
__int64 __fastcall sub_B49D00(__int64 a1)
{
  unsigned int v2; // eax
  _BYTE *v3; // rdi
  unsigned int v4; // r13d
  int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rdx
  bool v10; // r8
  int v11; // eax
  _QWORD v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v13[0] = *(_QWORD *)(a1 + 72);
  v2 = sub_A746F0(v13);
  v3 = *(_BYTE **)(a1 - 32);
  v4 = v2;
  if ( !*v3 )
  {
    v5 = sub_B2DC70((__int64)v3);
    if ( *(char *)(a1 + 7) < 0 )
    {
      v6 = sub_BD2BC0(a1);
      v8 = v6 + v7;
      v9 = 0;
      if ( *(char *)(a1 + 7) < 0 )
        v9 = sub_BD2BC0(a1);
      if ( (unsigned int)((v8 - v9) >> 4) )
      {
        if ( sub_B49990(a1) )
          v5 |= 0x55u;
        v10 = sub_B49A80(a1);
        v11 = v5;
        if ( v10 )
        {
          LOBYTE(v11) = v5 | 0xAA;
          v5 = v11;
        }
      }
    }
    v4 &= v5;
  }
  return v4;
}

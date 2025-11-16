// Function: sub_2E89ED0
// Address: 0x2e89ed0
//
__int64 __fastcall sub_2E89ED0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r8
  unsigned __int8 v6; // al
  unsigned int v7; // edx
  __int64 result; // rax

  v3 = *(_QWORD *)(a1 + 32);
  v4 = v3 + 40LL * a2;
  v5 = v3 + 40LL * a3;
  v6 = a2 + 1;
  if ( a2 >= 0xFF )
    v6 = -1;
  v7 = a3 + 1;
  *(_WORD *)(v5 + 2) = *(_WORD *)(v5 + 2) & 0xF00F | (16 * v6);
  if ( v7 > 0xFF )
    LOBYTE(v7) = -1;
  result = *(_WORD *)(v4 + 2) & 0xF00F;
  *(_WORD *)(v4 + 2) = result | (16 * (unsigned __int8)v7);
  return result;
}

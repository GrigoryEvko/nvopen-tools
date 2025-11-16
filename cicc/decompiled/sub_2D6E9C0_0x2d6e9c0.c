// Function: sub_2D6E9C0
// Address: 0x2d6e9c0
//
__int64 __fastcall sub_2D6E9C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // esi
  int v5; // eax
  int v6; // eax
  __int64 result; // rax
  _QWORD v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_DWORD *)(a1 + 16);
  v8[0] = a3;
  ++*(_QWORD *)a1;
  v6 = v5 + 1;
  if ( 4 * v6 >= 3 * v4 )
  {
    v4 *= 2;
  }
  else if ( v4 - *(_DWORD *)(a1 + 20) - v6 > v4 >> 3 )
  {
    goto LABEL_3;
  }
  sub_2D6E640(a1, v4);
  sub_2D67BB0(a1, a2, v8);
  v6 = *(_DWORD *)(a1 + 16) + 1;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v6;
  result = v8[0];
  if ( *(_QWORD *)(v8[0] + 16LL) != -4096 )
    --*(_DWORD *)(a1 + 20);
  return result;
}

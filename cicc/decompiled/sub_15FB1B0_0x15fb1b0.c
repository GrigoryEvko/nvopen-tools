// Function: sub_15FB1B0
// Address: 0x15fb1b0
//
__int64 __fastcall sub_15FB1B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rdx
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  __int64 result; // rax

  v2 = *(_QWORD *)(a2 - 24);
  sub_15F1EA0(a1, *(_QWORD *)a2, 62, a1 - 24, 1, 0);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v3 = *(_QWORD *)(a1 - 16);
    v4 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v4 = v3;
    if ( v3 )
      *(_QWORD *)(v3 + 16) = *(_QWORD *)(v3 + 16) & 3LL | v4;
  }
  *(_QWORD *)(a1 - 24) = v2;
  if ( v2 )
  {
    v5 = *(_QWORD *)(v2 + 8);
    *(_QWORD *)(a1 - 16) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = (a1 - 16) | *(_QWORD *)(v5 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (v2 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(v2 + 8) = a1 - 24;
  }
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 64) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 64) )
    sub_15F4C60(a1 + 56, a2 + 56);
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}

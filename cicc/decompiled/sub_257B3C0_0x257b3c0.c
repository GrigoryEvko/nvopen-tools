// Function: sub_257B3C0
// Address: 0x257b3c0
//
__int64 __fastcall sub_257B3C0(__int64 *a1, __int64 a2)
{
  _BYTE *v2; // rbx
  __int64 v4; // r13
  __int64 v5; // rax
  unsigned __int64 v6[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(_BYTE **)(a2 - 32);
  if ( !v2 || *v2 || (v2[33] & 0x20) != 0 )
    return 0;
  if ( sub_B2FC80(*(_QWORD *)(a2 - 32)) )
    return (unsigned int)sub_B2D610((__int64)v2, 6) ^ 1;
  v4 = *a1;
  sub_250D230(v6, (unsigned __int64)v2, 4, 0);
  v5 = sub_257B000(v4, v6[0], v6[1], a1[1], 0, 0, 1);
  if ( !v5 )
    return 0;
  return *(unsigned __int8 *)(v5 + 97);
}

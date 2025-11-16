// Function: sub_3026070
// Address: 0x3026070
//
__int64 __fastcall sub_3026070(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rax
  __int64 v5; // r12
  char v6; // al
  __int64 v7; // rax

  v4 = (_BYTE *)sub_B325F0(a2);
  if ( !v4 || (v5 = (__int64)v4, *v4) || (unsigned __int8)sub_CE9220((__int64)v4) || sub_B2FC80(v5) )
    sub_C64ED0("NVPTX aliasee must be a non-kernel function definition", 1u);
  v6 = *(_BYTE *)(a2 + 32) & 0xF;
  if ( ((v6 + 15) & 0xFu) <= 4 || v6 == 10 )
    sub_C64ED0("NVPTX aliasee must not be '.weak'", 1u);
  v7 = sub_31DB510(a1, a2);
  return sub_3025D20(a1, v5, v7, a3);
}

// Function: sub_11E0870
// Address: 0x11e0870
//
__int64 __fastcall sub_11E0870(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // r12
  __int64 *v5; // rax
  int v6; // [rsp-2Ch] [rbp-2Ch] BYREF

  if ( **(_BYTE **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) == 20 )
  {
    v3 = (__int64 *)sub_BD5C60(a2);
    v6 = 0;
    v4 = sub_A77AD0(v3, 0);
    v5 = (__int64 *)sub_BD5C60(a2);
    *(_QWORD *)(a2 + 72) = sub_A7B660((__int64 *)(a2 + 72), v5, &v6, 1, v4);
  }
  return 0;
}

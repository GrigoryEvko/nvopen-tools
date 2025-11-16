// Function: sub_1280D90
// Address: 0x1280d90
//
__int64 __fastcall sub_1280D90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rax

  v3 = a3 + 36;
  v5 = *(_QWORD *)(a3 + 56);
  if ( *(_BYTE *)(v5 + 173) != 2 )
    sub_127B550("cannot generate l-value for this constant!", (_DWORD *)(a3 + 36), 1);
  v6 = sub_126A1B0(*(__int64 **)(a2 + 32), v5, 0);
  v7 = sub_1289750(a2, v6, v3);
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = v7;
  *(_DWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}

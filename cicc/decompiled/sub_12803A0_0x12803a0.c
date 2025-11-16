// Function: sub_12803A0
// Address: 0x12803a0
//
__int64 __fastcall sub_12803A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // rax

  v5 = *(_QWORD *)(a3 + 56);
  if ( (*(_BYTE *)(v5 + 197) & 0x60) != 0 && *(_QWORD *)(v5 + 128) )
    v5 = *(_QWORD *)(v5 + 128);
  if ( dword_4D046EC )
    sub_127C010(v5, (_DWORD *)(a3 + 36));
  v6 = sub_1276020(*(_QWORD *)(a2 + 32), v5, 0, a4, a5);
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = v6;
  *(_DWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}

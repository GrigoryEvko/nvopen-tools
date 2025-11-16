// Function: sub_9228E0
// Address: 0x9228e0
//
__int64 __fastcall sub_9228E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx

  v3 = *(_QWORD *)(a3 + 56);
  if ( (*(_BYTE *)(v3 + 197) & 0x60) != 0 && *(_QWORD *)(v3 + 128) )
    v3 = *(_QWORD *)(v3 + 128);
  if ( dword_4D046EC )
    sub_91C360(v3, (_DWORD *)(a3 + 36));
  v4 = sub_917010(*(_QWORD *)(a2 + 32), v3, 0);
  v5 = *(_QWORD *)(v3 + 152);
  *(_DWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = v4;
  *(_QWORD *)(a1 + 16) = v5;
  *(_DWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  return a1;
}

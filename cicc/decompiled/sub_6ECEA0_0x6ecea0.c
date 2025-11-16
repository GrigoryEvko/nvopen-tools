// Function: sub_6ECEA0
// Address: 0x6ecea0
//
__int64 __fastcall sub_6ECEA0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rax

  if ( !sub_6ECD10(a1, (__int64)a2, a3, a4, a5, a6) )
    return 0;
  if ( unk_4D048C0 )
  {
    v7 = *(_QWORD *)(a1 + 144);
    if ( *(_BYTE *)(v7 + 24) == 3 )
    {
      v9 = *(_QWORD *)(v7 + 56);
      if ( *(_BYTE *)(v9 + 177) == 5 )
        v7 = *(_QWORD *)(v9 + 184);
    }
    if ( (unsigned int)sub_6DEAC0(v7)
      && (unsigned int)sub_6ECD90(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 72) + 16LL) + 56LL)) )
    {
      if ( sub_6E53E0(5, 0x8Bu, (_DWORD *)(a1 + 68)) )
        sub_684B30(0x8Bu, (_DWORD *)(a1 + 68));
      return 0;
    }
  }
  if ( (unsigned int)sub_6E5430() )
    sub_6851C0(0x8Bu, a2);
  sub_6E6840(a1);
  return 1;
}

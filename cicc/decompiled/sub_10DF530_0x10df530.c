// Function: sub_10DF530
// Address: 0x10df530
//
bool __fastcall sub_10DF530(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = *(_QWORD *)(a2 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  return (*(_DWORD *)(v2 + 36) & 0xFFFFFFFD) == 373;
}

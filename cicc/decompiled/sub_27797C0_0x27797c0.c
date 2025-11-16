// Function: sub_27797C0
// Address: 0x27797c0
//
__int64 __fastcall sub_27797C0(__int64 a1)
{
  __int64 v1; // rax
  int v2; // eax

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    goto LABEL_8;
  v2 = *(_DWORD *)(v1 + 36);
  if ( v2 != 228 )
  {
    if ( v2 == 230 )
      return *(_QWORD *)(a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
LABEL_8:
    BUG();
  }
  return *(_QWORD *)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
}

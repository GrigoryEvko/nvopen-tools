// Function: sub_72B540
// Address: 0x72b540
//
void __fastcall sub_72B540(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  if ( *(_BYTE *)(a1 + 24) == 1 && (unsigned __int8)(*(_BYTE *)(a1 + 56) - 105) <= 4u )
  {
    v2 = sub_72B0F0(*(_QWORD *)(a1 + 72), 0);
    if ( !v2 || (*(_BYTE *)(v2 + 194) & 0xE) == 0 )
    {
      *(_DWORD *)(a2 + 80) = 1;
      *(_DWORD *)(a2 + 72) = 1;
    }
  }
}

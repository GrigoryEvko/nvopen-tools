// Function: sub_D5F5C0
// Address: 0xd5f5c0
//
__int64 __fastcall sub_D5F5C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = *(_QWORD *)(a2 - 32);
  if ( v2 && !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a2 + 80) && *(_DWORD *)(v2 + 36) )
    return sub_D5EFE0(a1, a2);
  else
    return sub_D5EFE0(a1, a2);
}

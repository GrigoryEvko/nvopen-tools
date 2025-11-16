// Function: sub_B5A250
// Address: 0xb5a250
//
__int64 __fastcall sub_B5A250(__int64 a1)
{
  __int64 v1; // rax
  __int64 result; // rax
  __int64 v3; // [rsp+8h] [rbp-18h]

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v3 = sub_B5A1E0(*(_DWORD *)(v1 + 36));
  result = 0;
  if ( BYTE4(v3) )
    return *(_QWORD *)(a1 + 32 * ((unsigned int)v3 - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  return result;
}

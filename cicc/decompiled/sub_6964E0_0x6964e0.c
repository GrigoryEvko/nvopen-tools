// Function: sub_6964E0
// Address: 0x6964e0
//
__int64 __fastcall sub_6964E0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r12
  __int64 v6; // rdi
  __int64 v7; // rax

  if ( !dword_4F077C0 )
    return 0;
  v4 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(v4 + 25) == 2 )
  {
    if ( *(_BYTE *)(v4 + 24) != 2 )
      return 0;
  }
  else if ( !(unsigned int)sub_6ED0A0(v4 + 8) || *(_BYTE *)(v4 + 24) != 2 )
  {
    return 0;
  }
  v6 = *(_QWORD *)(v4 + 8);
  if ( v6 == a2 || (unsigned int)sub_8DED30(v6, a2, 3) )
  {
    v7 = sub_724D50(0);
    *a3 = v7;
    sub_72A510(v4 + 152, v7);
    return 1;
  }
  return 0;
}

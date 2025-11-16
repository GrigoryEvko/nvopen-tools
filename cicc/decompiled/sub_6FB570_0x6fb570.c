// Function: sub_6FB570
// Address: 0x6fb570
//
__int64 __fastcall sub_6FB570(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  char v7; // al

  result = sub_8D3410(*(_QWORD *)a1);
  if ( !(_DWORD)result )
    return result;
  v7 = *(_BYTE *)(a1 + 17);
  if ( v7 == 1 )
  {
    if ( !sub_6ED0A0(a1) )
      return sub_6FB030(a1, a2, v3, v4, v5, v6);
    v7 = *(_BYTE *)(a1 + 17);
  }
  if ( v7 == 2 || (result = sub_6ED0A0(a1), (_DWORD)result) )
  {
    if ( dword_4F077C4 == 2 )
      return sub_6FB030(a1, a2, v3, v4, v5, v6);
    if ( unk_4F07778 > 199900 )
      return sub_6FB030(a1, a2, v3, v4, v5, v6);
    result = dword_4F077C0;
    if ( dword_4F077C0 )
      return sub_6FB030(a1, a2, v3, v4, v5, v6);
  }
  return result;
}

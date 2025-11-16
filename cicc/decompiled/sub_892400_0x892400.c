// Function: sub_892400
// Address: 0x892400
//
__int64 __fastcall sub_892400(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 + 88);
  if ( !v1 || (*(_BYTE *)(a1 + 160) & 1) != 0 )
    return a1;
  else
    return *(_QWORD *)(v1 + 88);
}

// Function: sub_291D8A0
// Address: 0x291d8a0
//
__int64 __fastcall sub_291D8A0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 - 32);
  if ( !v1 || *(_BYTE *)v1 || *(_QWORD *)(v1 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  if ( *(_DWORD *)(v1 + 36) == 68 )
    return sub_B595C0(a1);
  else
    return sub_B58EB0(a1, 0);
}

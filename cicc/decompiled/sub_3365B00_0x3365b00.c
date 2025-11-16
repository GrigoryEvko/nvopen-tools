// Function: sub_3365B00
// Address: 0x3365b00
//
__int64 __fastcall sub_3365B00(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // rax

  if ( !a1 )
LABEL_8:
    BUG();
  while ( 1 )
  {
    v1 = *(_QWORD *)(a1 + 24);
    v2 = *(_QWORD *)(v1 - 32);
    if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(v1 + 80) || *(_DWORD *)(v2 + 36) != 16 )
      return *(_QWORD *)(a1 + 24);
    a1 = *(_QWORD *)(a1 + 8);
    if ( !a1 )
      goto LABEL_8;
  }
}

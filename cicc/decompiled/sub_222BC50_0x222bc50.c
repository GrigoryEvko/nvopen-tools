// Function: sub_222BC50
// Address: 0x222bc50
//
void __fastcall sub_222BC50(__int64 a1)
{
  unsigned __int64 v2; // rdi
  unsigned __int64 v3; // rdi

  if ( *(_BYTE *)(a1 + 168) )
  {
    v2 = *(_QWORD *)(a1 + 152);
    if ( v2 )
      j_j___libc_free_0_0(v2);
    *(_QWORD *)(a1 + 152) = 0;
    *(_BYTE *)(a1 + 168) = 0;
  }
  v3 = *(_QWORD *)(a1 + 208);
  if ( v3 )
    j_j___libc_free_0_0(v3);
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
}

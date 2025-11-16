// Function: sub_1313270
// Address: 0x1313270
//
void __fastcall sub_1313270(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rax

  if ( *(_BYTE *)(a1 + 816) )
    goto LABEL_7;
  v1 = *(_QWORD *)(a1 + 16);
  if ( v1 >= 0xFFFFFFFFFFFFF001LL )
    v1 = 0;
  *(_QWORD *)(a1 + 832) = v1;
  v2 = *(_QWORD *)(a1 + 32);
  if ( v2 >= 0xFFFFFFFFFFFFF001LL )
    v2 = 0;
  *(_QWORD *)(a1 + 848) = v2;
  _mm_mfence();
  if ( *(_BYTE *)(a1 + 816) )
  {
LABEL_7:
    *(_QWORD *)(a1 + 832) = 0;
    *(_QWORD *)(a1 + 848) = 0;
  }
}

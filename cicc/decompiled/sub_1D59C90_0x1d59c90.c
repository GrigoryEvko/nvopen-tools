// Function: sub_1D59C90
// Address: 0x1d59c90
//
__int64 *__fastcall sub_1D59C90(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    sub_1D91830(v1);
    *(_BYTE *)(v2 + 40) = 1;
    *(_QWORD *)v2 = &off_4985500;
    *(_DWORD *)(v2 + 44) = 0;
    *(_WORD *)(v2 + 50) = 0;
    *(_BYTE *)(v2 + 52) = 0;
  }
  *a1 = v2;
  return a1;
}

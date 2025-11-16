// Function: sub_230D800
// Address: 0x230d800
//
__int64 *__fastcall sub_230D800(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx

  v1 = sub_22077B0(0x28u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 1;
    *(_QWORD *)(v1 + 16) = 0;
    *(_QWORD *)(v1 + 24) = 0;
    *(_QWORD *)v1 = &unk_4A0B380;
    *(_DWORD *)(v1 + 32) = 0;
  }
  sub_C7D6A0(0, 0, 8);
  *a1 = v2;
  sub_C7D6A0(0, 0, 8);
  return a1;
}

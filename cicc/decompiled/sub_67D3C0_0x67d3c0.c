// Function: sub_67D3C0
// Address: 0x67d3c0
//
_BOOL8 __fastcall sub_67D3C0(int *a1, char a2, _DWORD *a3)
{
  char v3; // al
  char v5[4]; // [rsp+Ch] [rbp-4h] BYREF

  v3 = a2;
  v5[0] = a2;
  if ( !unk_4F0772C && (unsigned __int8)a2 <= 7u )
  {
    sub_67C4B0(a1, v5, a3);
    v3 = v5[0];
  }
  return (unsigned __int8)v3 > 6u;
}

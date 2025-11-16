// Function: sub_160CFB0
// Address: 0x160cfb0
//
__int64 sub_160CFB0()
{
  __int64 v0; // rax
  __int64 v1; // r12

  v0 = sub_22077B0(16);
  v1 = v0;
  if ( v0 )
  {
    sub_16C3010(v0, 1);
    *(_BYTE *)(v1 + 12) = 1;
    *(_DWORD *)(v1 + 8) = 0;
  }
  return v1;
}

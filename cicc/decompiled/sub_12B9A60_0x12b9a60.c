// Function: sub_12B9A60
// Address: 0x12b9a60
//
__int64 sub_12B9A60()
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

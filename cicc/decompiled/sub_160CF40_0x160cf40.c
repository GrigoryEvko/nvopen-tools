// Function: sub_160CF40
// Address: 0x160cf40
//
__int64 sub_160CF40()
{
  __int64 v0; // rax
  __int64 v1; // r12

  v0 = sub_22077B0(144);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)v0 = 0;
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = 0;
    *(_DWORD *)(v0 + 24) = 0;
    sub_16D7E20(v0 + 32, "pass", 4, "... Pass execution timing report ...", 36);
  }
  return v1;
}

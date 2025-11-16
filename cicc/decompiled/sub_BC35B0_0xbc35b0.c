// Function: sub_BC35B0
// Address: 0xbc35b0
//
__int64 sub_BC35B0()
{
  __int64 v0; // rax
  __int64 v1; // r12

  v0 = sub_22077B0(168);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)v0 = 0;
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = 0x1000000000LL;
    *(_QWORD *)(v0 + 24) = 0;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_DWORD *)(v0 + 48) = 0;
    sub_C9E8E0(v0 + 56, "pass", 4, "Pass execution timing report", 28);
  }
  return v1;
}

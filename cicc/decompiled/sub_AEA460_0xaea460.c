// Function: sub_AEA460
// Address: 0xaea460
//
__int64 __fastcall sub_AEA460(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  unsigned int v4; // r8d
  int v5; // eax

  v1 = sub_BA91D0(a1, "debug-info-assignment-tracking", 30);
  v4 = 0;
  if ( v1 )
  {
    LOBYTE(v5) = sub_AD7890(*(_QWORD *)(v1 + 136), (__int64)"debug-info-assignment-tracking", v2, v3, 0);
    return v5 ^ 1u;
  }
  return v4;
}

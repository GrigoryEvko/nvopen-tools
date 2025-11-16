// Function: sub_21635B0
// Address: 0x21635b0
//
__int64 __fastcall sub_21635B0(__int64 a1, __int64 a2)
{
  int v2; // ebx
  unsigned int v3; // ebx
  void *v4; // r13
  __int64 v6; // rax

  v2 = *(_DWORD *)(a2 + 16);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = v2;
  v3 = (unsigned int)(v2 + 63) >> 6;
  v4 = (void *)malloc(8LL * v3);
  if ( !v4 )
  {
    if ( 8LL * v3 || (v6 = malloc(1u)) == 0 )
      sub_16BD1C0("Allocation failed", 1u);
    else
      v4 = (void *)v6;
  }
  *(_QWORD *)a1 = v4;
  *(_QWORD *)(a1 + 8) = v3;
  if ( v3 )
    memset(v4, 0, 8LL * v3);
  return a1;
}

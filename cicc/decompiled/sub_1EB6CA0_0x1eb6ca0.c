// Function: sub_1EB6CA0
// Address: 0x1eb6ca0
//
__int64 __fastcall sub_1EB6CA0(__int64 a1)
{
  unsigned __int64 v1; // r13
  _QWORD *v2; // rax

  v1 = malloc(8u);
  if ( !v1 )
    sub_16BD1C0("Allocation failed", 1u);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 8;
  v2 = (_QWORD *)malloc(8u);
  if ( !v2 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v2 = 0;
  }
  *(_QWORD *)a1 = v2;
  *(_QWORD *)(a1 + 8) = 1;
  *v2 = 2;
  _libc_free(v1);
  return a1;
}

// Function: sub_16CCCB0
// Address: 0x16cccb0
//
__int64 __fastcall sub_16CCCB0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v8; // rax

  *a1 = 0;
  v4 = *(_QWORD *)(a3 + 8);
  a1[1] = a2;
  if ( *(_QWORD *)(a3 + 16) == v4 )
  {
    a1[2] = a2;
  }
  else
  {
    v5 = 8LL * *(unsigned int *)(a3 + 24);
    v6 = malloc(v5);
    if ( !v6 )
    {
      if ( v5 || (v8 = malloc(1u)) == 0 )
        sub_16BD1C0("Allocation failed", 1u);
      else
        v6 = v8;
    }
    a1[2] = v6;
  }
  return sub_16CCC50((__int64)a1, a3);
}

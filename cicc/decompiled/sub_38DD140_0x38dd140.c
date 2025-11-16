// Function: sub_38DD140
// Address: 0x38dd140
//
__int64 __fastcall sub_38DD140(__int64 a1)
{
  __int64 v2; // rdi
  const char *v3; // [rsp+0h] [rbp-30h] BYREF
  char v4; // [rsp+10h] [rbp-20h]
  char v5; // [rsp+11h] [rbp-1Fh]

  if ( sub_38DD120(a1) )
    return *(_QWORD *)(a1 + 32) - 80LL;
  v2 = *(_QWORD *)(a1 + 8);
  v5 = 1;
  v4 = 3;
  v3 = "this directive must appear between .cfi_startproc and .cfi_endproc directives";
  sub_38BE3D0(v2, 0, (__int64)&v3);
  return 0;
}

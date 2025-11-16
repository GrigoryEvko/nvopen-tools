// Function: sub_E99320
// Address: 0xe99320
//
__int64 __fastcall sub_E99320(__int64 a1)
{
  _QWORD *v2; // rsi
  __int64 v3; // rdi
  const char *v4; // [rsp+0h] [rbp-40h] BYREF
  char v5; // [rsp+20h] [rbp-20h]
  char v6; // [rsp+21h] [rbp-1Fh]

  if ( sub_E99310(a1) )
    return *(_QWORD *)(a1 + 24) + 96LL * *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL * *(unsigned int *)(a1 + 56) - 16);
  v2 = *(_QWORD **)(a1 + 264);
  v3 = *(_QWORD *)(a1 + 8);
  v6 = 1;
  v4 = "this directive must appear between .cfi_startproc and .cfi_endproc directives";
  v5 = 3;
  if ( v2 )
    v2 = (_QWORD *)*v2;
  sub_E66880(v3, v2, (__int64)&v4);
  return 0;
}

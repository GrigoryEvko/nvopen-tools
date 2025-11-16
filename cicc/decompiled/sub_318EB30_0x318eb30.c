// Function: sub_318EB30
// Address: 0x318eb30
//
_QWORD *__fastcall sub_318EB30(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdx

  v2 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL);
  v3 = 0;
  if ( v2 )
    v3 = sub_3186770(*(_QWORD *)(a2 + 24), *(_QWORD *)(v2 + 24));
  v4 = *(_QWORD *)(a2 + 24);
  *a1 = v2;
  a1[1] = v3;
  a1[2] = v4;
  return a1;
}

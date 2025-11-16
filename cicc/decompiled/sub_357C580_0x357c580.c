// Function: sub_357C580
// Address: 0x357c580
//
__int64 __fastcall sub_357C580(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 36;
    *(_QWORD *)v1 = "Rename Register Operands Canonically";
    *(_QWORD *)(v1 + 16) = "mir-canonicalizer";
    *(_QWORD *)(v1 + 32) = &unk_503F098;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_357C470;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

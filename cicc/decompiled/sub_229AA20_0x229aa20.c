// Function: sub_229AA20
// Address: 0x229aa20
//
__int64 __fastcall sub_229AA20(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 76;
    *(_QWORD *)v1 = "Print postdominance tree of function to 'dot' file (with no function bodies)";
    *(_QWORD *)(v1 + 16) = "dot-postdom-only";
    *(_QWORD *)(v1 + 32) = &unk_4FDB5EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_229C960;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

// Function: sub_2285950
// Address: 0x2285950
//
__int64 __fastcall sub_2285950(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 30;
    *(_QWORD *)v1 = "Print call graph to 'dot' file";
    *(_QWORD *)(v1 + 16) = "dot-callgraph";
    *(_QWORD *)(v1 + 32) = &unk_4FDAEAC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_22857F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

// Function: sub_97DFA0
// Address: 0x97dfa0
//
__int64 __fastcall sub_97DFA0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Target Library Information";
    *(_QWORD *)(v1 + 16) = "targetlibinfo";
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 32) = &unk_4F6D3F0;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_980230;
  }
  sub_BC3090(rwlock);
  return v2;
}

// Function: sub_37F1A50
// Address: 0x37f1a50
//
__int64 __fastcall sub_37F1A50(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E6D3E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 39;
    *(_QWORD *)v1 = "Machine Dominance Frontier Construction";
    *(_QWORD *)(v1 + 16) = "machine-domfrontier";
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 32) = &unk_505150C;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_37F1EC0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

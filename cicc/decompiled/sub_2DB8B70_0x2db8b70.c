// Function: sub_2DB8B70
// Address: 0x2db8b70
//
__int64 __fastcall sub_2DB8B70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Bundle Machine CFG Edges";
    *(_QWORD *)(v1 + 16) = "edge-bundles";
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 32) = &unk_501D134;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_2DB8A60;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

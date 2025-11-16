// Function: sub_37F4310
// Address: 0x37f4310
//
__int64 __fastcall sub_37F4310(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "ReachingDefAnalysis";
    *(_QWORD *)(v1 + 16) = "reaching-defs-analysis";
    *(_QWORD *)(v1 + 24) = 22;
    *(_QWORD *)(v1 + 32) = &unk_5051514;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_37F4D60;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

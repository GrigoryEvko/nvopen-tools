// Function: sub_3094240
// Address: 0x3094240
//
__int64 __fastcall sub_3094240(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2EA61A0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Optimize NVPTX ld.param";
    *(_QWORD *)(v1 + 16) = "param-opt";
    *(_QWORD *)(v1 + 32) = &unk_502D440;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 9;
    *(_QWORD *)(v1 + 48) = sub_3094500;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

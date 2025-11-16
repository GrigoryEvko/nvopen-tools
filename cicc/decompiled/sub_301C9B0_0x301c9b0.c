// Function: sub_301C9B0
// Address: 0x301c9b0
//
__int64 __fastcall sub_301C9B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2EA61A0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 15;
    *(_QWORD *)v1 = "Insert XRay ops";
    *(_QWORD *)(v1 + 16) = "xray-instrumentation";
    *(_QWORD *)(v1 + 32) = &unk_502A90C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_301D420;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

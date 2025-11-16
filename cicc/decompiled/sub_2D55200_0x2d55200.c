// Function: sub_2D55200
// Address: 0x2d55200
//
__int64 __fastcall sub_2D55200(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 14;
    *(_QWORD *)v1 = "Prepare callbr";
    *(_QWORD *)(v1 + 16) = "callbrprepare";
    *(_QWORD *)(v1 + 32) = &unk_5016964;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_2D55570;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

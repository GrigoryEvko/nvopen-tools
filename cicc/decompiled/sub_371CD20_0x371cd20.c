// Function: sub_371CD20
// Address: 0x371cd20
//
__int64 __fastcall sub_371CD20(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2C6F190((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Insert phi elim copies";
    *(_QWORD *)(v1 + 16) = "do-cssa";
    *(_QWORD *)(v1 + 24) = 7;
    *(_QWORD *)(v1 + 32) = &unk_505099C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_371DF50;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

// Function: sub_2C8FC50
// Address: 0x2c8fc50
//
__int64 __fastcall sub_2C8FC50(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_D9A960((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_31C5560(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 43;
    *(_QWORD *)v1 = "Optimize address mode with Scalar Evolution";
    *(_QWORD *)(v1 + 16) = "codegenpreparescev";
    *(_QWORD *)(v1 + 32) = &unk_5011B54;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_2C946E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

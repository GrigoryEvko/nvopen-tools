// Function: sub_2FC9200
// Address: 0x2fc9200
//
__int64 __fastcall sub_2FC9200(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2FEF6D0(rwlock);
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Insert stack protectors";
    *(_QWORD *)(v1 + 16) = "stack-protector";
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 32) = &unk_502608C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2FC9E00;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

// Function: sub_35671A0
// Address: 0x35671a0
//
__int64 __fastcall sub_35671A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E6D3E0((__int64)rwlock);
  sub_2EB3F30((__int64)rwlock);
  sub_37F1CF0(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 39;
    *(_QWORD *)v1 = "Detect single entry single exit regions";
    *(_QWORD *)(v1 + 16) = "machine-region-info";
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 32) = &unk_503EEEC;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_35689D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

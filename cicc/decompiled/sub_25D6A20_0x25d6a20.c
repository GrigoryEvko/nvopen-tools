// Function: sub_25D6A20
// Address: 0x25d6a20
//
__int64 __fastcall sub_25D6A20(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Dead Global Elimination";
    *(_QWORD *)(v1 + 16) = "globaldce";
    *(_QWORD *)(v1 + 32) = &unk_4FF14EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 9;
    *(_QWORD *)(v1 + 48) = sub_25D6E30;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

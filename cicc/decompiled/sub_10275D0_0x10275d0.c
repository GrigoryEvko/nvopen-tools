// Function: sub_10275D0
// Address: 0x10275d0
//
__int64 __fastcall sub_10275D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_10284E0();
  sub_D4AA90((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 29;
    *(_QWORD *)v1 = "Lazy Block Frequency Analysis";
    *(_QWORD *)(v1 + 16) = "lazy-block-freq";
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 32) = &unk_4F8EE48;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_10279F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

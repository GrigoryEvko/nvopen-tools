// Function: sub_2FAEBD0
// Address: 0x2faebd0
//
__int64 __fastcall sub_2FAEBD0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2DB8D80((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 29;
    *(_QWORD *)v1 = "Spill Code Placement Analysis";
    *(_QWORD *)(v1 + 16) = "spill-code-placement";
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 32) = &unk_5025C34;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_2FAEDD0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

// Function: sub_D75D40
// Address: 0xd75d40
//
__int64 __fastcall sub_D75D40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_FDC5A0();
  sub_D84940(rwlock);
  sub_D8A010(rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Module Summary Analysis";
    *(_QWORD *)(v1 + 16) = "module-summary-analysis";
    *(_QWORD *)(v1 + 24) = 23;
    *(_QWORD *)(v1 + 32) = &unk_4F87814;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_D783B0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

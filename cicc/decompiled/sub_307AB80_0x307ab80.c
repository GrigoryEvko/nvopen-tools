// Function: sub_307AB80
// Address: 0x307ab80
//
__int64 __fastcall sub_307AB80(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_307BEB0();
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 30;
    *(_QWORD *)v1 = "Machine Function Extra Printer";
    *(_QWORD *)(v1 + 16) = "extra-machineinstr-printer";
    *(_QWORD *)(v1 + 24) = 26;
    *(_QWORD *)(v1 + 32) = &unk_502D264;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_307AC10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

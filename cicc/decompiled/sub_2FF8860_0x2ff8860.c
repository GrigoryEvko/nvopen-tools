// Function: sub_2FF8860
// Address: 0x2ff8860
//
__int64 __fastcall sub_2FF8860(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 28;
    *(_QWORD *)v1 = "Two-Address instruction pass";
    *(_QWORD *)(v1 + 16) = "twoaddressinstruction";
    *(_QWORD *)(v1 + 32) = &unk_502A48C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 48) = sub_2FFA440;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

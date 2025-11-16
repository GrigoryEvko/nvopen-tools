// Function: sub_2617510
// Address: 0x2617510
//
__int64 __fastcall sub_2617510(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 44;
    *(_QWORD *)v1 = "Extract at most one loop into a new function";
    *(_QWORD *)(v1 + 16) = "loop-extract-single";
    *(_QWORD *)(v1 + 32) = &unk_4FF2A4C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 48) = sub_26181E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

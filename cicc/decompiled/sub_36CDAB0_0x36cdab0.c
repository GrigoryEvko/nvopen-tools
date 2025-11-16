// Function: sub_36CDAB0
// Address: 0x36cdab0
//
__int64 __fastcall sub_36CDAB0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 40;
    *(_QWORD *)v1 = "NVPTX Address space based Alias Analysis";
    *(_QWORD *)(v1 + 16) = "nvptx-aa";
    *(_QWORD *)(v1 + 24) = 8;
    *(_QWORD *)(v1 + 32) = &unk_5040919;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_36CEF60;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}

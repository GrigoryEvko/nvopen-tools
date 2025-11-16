// Function: sub_1688F80
// Address: 0x1688f80
//
__int64 sub_1688F80()
{
  __int64 v0; // r12
  __int64 v1; // rax

  if ( !unk_4F9F820 )
    sub_1688CB0();
  v0 = malloc(0x118u);
  if ( !v0 )
    sub_16863E0((__int64)&unk_4CD28F0);
  memset((void *)v0, 0, 0x118u);
  pthread_cond_init((pthread_cond_t *)(v0 + 128), 0);
  pthread_mutex_init((pthread_mutex_t *)(v0 + 176), 0);
  sem_init((sem_t *)(v0 + 216), 0, 0);
  sub_1688E30();
  v1 = unk_4F9F820;
  *(_QWORD *)(v0 + 264) = &unk_4F9F720;
  unk_4F9F820 = v0;
  *(_QWORD *)(v0 + 256) = v1;
  *(_QWORD *)(v1 + 264) = v0;
  sub_1688E70();
  return v0;
}

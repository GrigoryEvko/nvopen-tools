// Function: sub_1688E80
// Address: 0x1688e80
//
int __fastcall sub_1688E80(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v3; // rdi
  __int64 v4; // rax
  sem_t *v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rdx
  int result; // eax

  v1 = *(_QWORD *)(a1 + 256);
  v3 = a1 + 128;
  v4 = *(_QWORD *)(v3 + 136);
  v5 = *(sem_t **)(v3 + 120);
  *(_QWORD *)(v1 + 264) = v4;
  *(_QWORD *)(v4 + 256) = *(_QWORD *)(v3 + 128);
  pthread_cond_destroy((pthread_cond_t *)v3);
  pthread_mutex_destroy((pthread_mutex_t *)(a1 + 176));
  sem_destroy((sem_t *)(a1 + 216));
  *(_BYTE *)(a1 + 272) = 1;
  v6 = unk_4F9F5E0;
  *(_QWORD *)(a1 + 264) = &unk_4F9F4E0;
  *(_QWORD *)(a1 + 256) = v6;
  v7 = unk_4F9F5E0;
  unk_4F9F5E0 = a1;
  *(_QWORD *)(v7 + 264) = a1;
  result = sub_1688E70();
  if ( v5 )
    return sem_post(v5);
  return result;
}

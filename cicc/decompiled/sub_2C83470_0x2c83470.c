// Function: sub_2C83470
// Address: 0x2c83470
//
__int64 __fastcall sub_2C83470(__int64 a1, int a2, __int64 (*a3)(), char *a4)
{
  pthread_mutex_t *v7; // rbx
  unsigned int v8; // eax
  __int64 *v9; // rax
  unsigned int *v10; // rcx
  __int64 v11; // r8
  size_t v12; // rax
  unsigned int v13; // r13d
  _QWORD *v15; // rax
  __int64 v16; // rdi
  _QWORD *v17; // r8
  __int64 *v18; // [rsp+8h] [rbp-38h]

  if ( !qword_5011170 )
    sub_C7D570((__int64 *)&qword_5011170, sub_B3AE10, (__int64)sub_B3AE40);
  v7 = qword_5011170;
  if ( &_pthread_key_create )
  {
    v8 = pthread_mutex_lock(qword_5011170);
    if ( v8 )
      sub_4264C5(v8);
  }
  if ( *(_BYTE *)a1 )
  {
    v15 = (_QWORD *)sub_22077B0(0x20u);
    v16 = (__int64)v15;
    if ( v15 )
    {
      v15[1] = 0;
      v15[2] = 0;
      v15[3] = 0;
      *v15 = &unk_4A250F8;
    }
    v17 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)(a1 + 8) = v15;
    if ( v17 )
    {
      sub_2C83430(v17);
      v16 = *(_QWORD *)(a1 + 8);
    }
    sub_C53270(v16);
  }
  v9 = sub_CEADF0();
  v10 = 0;
  v11 = (__int64)v9;
  if ( a4 )
  {
    v18 = v9;
    v12 = strlen(a4);
    v11 = (__int64)v18;
    v10 = (unsigned int *)v12;
  }
  v13 = sub_C5AF60(a2, a3, (__int64 *)a4, v10, v11, 0, 0);
  sub_2C83460(a1);
  if ( &_pthread_key_create )
    pthread_mutex_unlock(v7);
  return v13;
}

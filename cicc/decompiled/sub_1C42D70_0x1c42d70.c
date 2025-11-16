// Function: sub_1C42D70
// Address: 0x1c42d70
//
__int64 __fastcall sub_1C42D70(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // r14
  __int64 v4; // r13
  unsigned int v5; // r15d
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // r14
  __int64 v12; // rax
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // rdx
  __int64 *v16; // rdx
  __int64 v17; // [rsp+8h] [rbp-38h]
  __int64 v18; // [rsp+8h] [rbp-38h]

  if ( !byte_4FBB548 && (unsigned int)sub_2207590(&byte_4FBB548) )
  {
    qword_4FBB560 = 0;
    qword_4FBB570 = (__int64)&unk_4FBB580;
    qword_4FBB578 = 0x400000000LL;
    qword_4FBB568 = 0;
    qword_4FBB5A0 = (__int64)&qword_4FBB5B0;
    qword_4FBB5A8 = 0;
    qword_4FBB5B0 = 0;
    qword_4FBB5B8 = 1;
    __cxa_atexit((void (*)(void *))sub_1C42CD0, &qword_4FBB5B0 - 10, &qword_4A427C0);
    sub_2207640(&byte_4FBB548);
  }
  if ( &_pthread_key_create )
  {
    v2 = pthread_mutex_lock(&stru_4FBB5E0);
    if ( v2 )
      sub_4264C5(v2);
  }
  v3 = -a2;
  qword_4FBB5B0 += a1;
  if ( a1 + (-a2 & (unsigned __int64)(qword_4FBB560 + a2 - 1)) - qword_4FBB560 <= qword_4FBB568 - qword_4FBB560 )
  {
    v10 = -a2 & (qword_4FBB560 + a2 - 1);
    qword_4FBB560 = v10 + a1;
  }
  else
  {
    v4 = a2 - 1;
    if ( (unsigned __int64)(a1 + a2 - 1) > 0x1000 )
    {
      v12 = malloc(a1 + v4);
      if ( !v12 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v12 = 0;
      }
      v15 = (unsigned int)qword_4FBB5A8;
      if ( (unsigned int)qword_4FBB5A8 >= HIDWORD(qword_4FBB5A8) )
      {
        v18 = v12;
        sub_16CD150((__int64)(&qword_4FBB5B0 - 2), &qword_4FBB5B0, 0, 16, v13, v14);
        v15 = (unsigned int)qword_4FBB5A8;
        v12 = v18;
      }
      v16 = (__int64 *)(qword_4FBB5A0 + 16 * v15);
      *v16 = v12;
      v10 = (v12 + v4) & v3;
      v16[1] = a1 + a2 - 1;
      LODWORD(qword_4FBB5A8) = qword_4FBB5A8 + 1;
    }
    else
    {
      v5 = qword_4FBB578;
      v6 = 0x40000000000LL;
      if ( (unsigned int)qword_4FBB578 >> 7 < 0x1E )
        v6 = 4096LL << ((unsigned int)qword_4FBB578 >> 7);
      v7 = malloc(v6);
      if ( !v7 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v5 = qword_4FBB578;
        v7 = 0;
      }
      if ( HIDWORD(qword_4FBB578) <= v5 )
      {
        v17 = v7;
        sub_16CD150((__int64)&unk_4FBB580 - 16, &unk_4FBB580, 0, 8, v8, v9);
        v5 = qword_4FBB578;
        v7 = v17;
      }
      *(_QWORD *)(qword_4FBB570 + 8LL * v5) = v7;
      v10 = (v4 + v7) & v3;
      LODWORD(qword_4FBB578) = qword_4FBB578 + 1;
      qword_4FBB568 = v7 + v6;
      qword_4FBB560 = v10 + a1;
    }
  }
  if ( &_pthread_key_create )
    pthread_mutex_unlock(&stru_4FBB5E0);
  return v10;
}

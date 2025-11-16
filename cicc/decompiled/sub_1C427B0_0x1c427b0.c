// Function: sub_1C427B0
// Address: 0x1c427b0
//
__int64 __fastcall sub_1C427B0(__int64 a1, int a2, const void *a3, const char *a4, __int64 a5, __int64 a6)
{
  pthread_mutex_t **v9; // rbx
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  size_t v13; // rax
  unsigned int v14; // r13d
  _QWORD *v16; // rax
  __int64 v17; // rdi
  _QWORD *v18; // r8
  __int64 *v19; // [rsp+8h] [rbp-38h]

  if ( !qword_4FBB530 )
    sub_16C1EA0((__int64)&qword_4FBB530, sub_12B9A60, (__int64)sub_12B9AC0, (__int64)a4, a5, a6);
  v9 = (pthread_mutex_t **)qword_4FBB530;
  sub_16C30C0((pthread_mutex_t **)qword_4FBB530);
  if ( *(_BYTE *)a1 )
  {
    v16 = (_QWORD *)sub_22077B0(32);
    v17 = (__int64)v16;
    if ( v16 )
    {
      v16[1] = 0;
      v16[2] = 0;
      v16[3] = 0;
      *v16 = &unk_49F7B08;
    }
    v18 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)(a1 + 8) = v16;
    if ( v18 )
    {
      sub_1C42770(v18);
      v17 = *(_QWORD *)(a1 + 8);
    }
    sub_16B1F80(v17);
  }
  v10 = sub_1C3E840();
  v11 = 0;
  v12 = (__int64)v10;
  if ( a4 )
  {
    v19 = v10;
    v13 = strlen(a4);
    v12 = (__int64)v19;
    v11 = v13;
  }
  v14 = sub_16B7CB0(a2, a3, (__int64)a4, v11, v12);
  sub_1C427A0(a1);
  sub_16C30E0(v9);
  return v14;
}

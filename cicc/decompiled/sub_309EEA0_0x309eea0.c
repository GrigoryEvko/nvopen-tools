// Function: sub_309EEA0
// Address: 0x309eea0
//
__int64 __fastcall sub_309EEA0(__int64 *a1, __int64 *a2)
{
  __int128 *v2; // r12
  __int64 v3; // rdx
  char *v4; // rcx
  unsigned int v5; // r12d
  __int64 *v7; // r13
  struct __jmp_buf_tag *v8; // r12
  int v9; // eax
  void *v10; // rdi
  unsigned __int64 v12[2]; // [rsp+10h] [rbp-80h] BYREF
  _BYTE v13[16]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v14[12]; // [rsp+30h] [rbp-60h] BYREF

  sub_CEAEC0();
  v2 = sub_BC2B00();
  sub_22EB2E0((__int64)v2);
  sub_31CE940(v2);
  v12[0] = (unsigned __int64)v13;
  v14[5] = 0x100000000LL;
  v12[1] = 0;
  v13[0] = 0;
  memset(&v14[1], 0, 32);
  v14[0] = &unk_49DD210;
  v14[6] = v12;
  sub_CB5980((__int64)v14, 0, 0, 0);
  if ( (unsigned __int8)sub_C09360(a1, (__int64)v14, 0) )
  {
    v5 = 6;
    sub_CEB590(v12, 1, v3, v4);
    sub_CEAF80(a2);
    goto LABEL_3;
  }
  v7 = sub_CEACC0();
  v8 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v7);
  if ( !v8 )
  {
    v10 = (void *)sub_CEECD0(200, 8u);
    memset(v10, 0, 0xC8u);
    sub_C94E10((__int64)v7, v10);
    v8 = (struct __jmp_buf_tag *)sub_C94E20((__int64)v7);
  }
  v9 = _setjmp(v8);
  if ( v9 )
  {
    if ( v9 == 1 )
    {
      v5 = 6;
      sub_CEAF80(a2);
      goto LABEL_3;
    }
  }
  else
  {
    sub_2C83050((char *)a1, 0, 0, 0, 0);
  }
  v5 = 0;
  sub_CEAF80(a2);
LABEL_3:
  v14[0] = &unk_49DD210;
  sub_CB5840((__int64)v14);
  if ( (_BYTE *)v12[0] != v13 )
    j_j___libc_free_0(v12[0]);
  return v5;
}

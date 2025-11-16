// Function: sub_A1B8B0
// Address: 0xa1b8b0
//
__int64 __fastcall sub_A1B8B0(__int64 *a1, unsigned int a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  __int64 v11; // r8
  __int64 v12; // rax
  _QWORD *v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // rdi
  unsigned int v18; // ebx
  __int64 v19; // rdi
  unsigned __int64 v21; // rdx
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+18h] [rbp-48h] BYREF
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  volatile signed __int32 *v25; // [rsp+28h] [rbp-38h]

  v8 = a3;
  sub_A19830(*a1, a2, 3u);
  v9 = (_QWORD *)sub_22077B0(544);
  v10 = v9;
  if ( v9 )
  {
    v11 = (__int64)(v9 + 2);
    v9[1] = 0x100000001LL;
    *v9 = &unk_49D9900;
    v9[2] = v9 + 4;
    v9[3] = 0x2000000000LL;
    v12 = 0;
  }
  else
  {
    v12 = MEMORY[0x18];
    v11 = 16;
    v21 = MEMORY[0x18] + 1LL;
    if ( v21 > MEMORY[0x1C] )
    {
      sub_C8D5F0(16, 32, v21, 16);
      v12 = MEMORY[0x18];
      v11 = 16;
    }
  }
  v13 = (_QWORD *)(v10[2] + 16 * v12);
  *v13 = v8;
  v13[1] = 1;
  v14 = *((unsigned int *)v10 + 7);
  v15 = (unsigned int)(*((_DWORD *)v10 + 6) + 1);
  *((_DWORD *)v10 + 6) = v15;
  if ( v15 + 1 > v14 )
  {
    v22 = v11;
    sub_C8D5F0(v11, v10 + 4, v15 + 1, 16);
    v15 = *((unsigned int *)v10 + 6);
    v11 = v22;
  }
  v16 = (_QWORD *)(v10[2] + 16 * v15);
  *v16 = 0;
  v16[1] = 10;
  v17 = *a1;
  ++*((_DWORD *)v10 + 6);
  v25 = (volatile signed __int32 *)v10;
  v24 = v11;
  v18 = sub_A1AB30(v17, &v24);
  if ( v25 )
    sub_A191D0(v25);
  v19 = *a1;
  v23 = v8;
  sub_A1B020(v19, v18, (__int64)&v23, 1, a4, a5, v24, 0);
  return sub_A192A0(*a1);
}

// Function: sub_1BF20B0
// Address: 0x1bf20b0
//
__int64 __fastcall sub_1BF20B0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rdi
  unsigned int v6; // ebx
  void *v8; // rax
  const void *v9; // rsi
  __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  void *v11; // [rsp+8h] [rbp-38h]
  __int64 v12; // [rsp+10h] [rbp-30h]
  unsigned int v13; // [rsp+18h] [rbp-28h]

  v4 = a1[6];
  v12 = 0;
  v10 = 0;
  v11 = 0;
  v13 = 0;
  if ( v4 )
  {
    j___libc_free_0(0);
    v5 = *(unsigned int *)(v4 + 88);
    v13 = v5;
    if ( (_DWORD)v5 )
    {
      v8 = (void *)sub_22077B0(16 * v5);
      v9 = *(const void **)(v4 + 72);
      v11 = v8;
      v12 = *(_QWORD *)(v4 + 80);
      memcpy(v8, v9, 16LL * v13);
    }
    else
    {
      v11 = 0;
      v12 = 0;
    }
  }
  v6 = sub_385E580(a1[2], a2, *a1, &v10, 1, 0);
  if ( ((v6 + 1) & 0xFFFFFFFD) != 0 )
    v6 = 0;
  j___libc_free_0(v11);
  return v6;
}

// Function: sub_2AB7CB0
// Address: 0x2ab7cb0
//
__int64 __fastcall sub_2AB7CB0(__int64 a1)
{
  unsigned __int64 v2; // r13
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r14
  __int64 v8; // rax
  __int64 v9; // rsi
  _QWORD *v10; // rbx
  _QWORD *v11; // r13
  _QWORD v12[5]; // [rsp+8h] [rbp-88h] BYREF
  void *v13; // [rsp+30h] [rbp-60h]
  _QWORD v14[11]; // [rsp+38h] [rbp-58h] BYREF

  v2 = *(_QWORD *)(a1 + 128);
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 40);
    if ( v3 != v2 + 56 )
      _libc_free(v3);
    j_j___libc_free_0(v2);
  }
  if ( *(_BYTE *)(a1 + 96) )
  {
    v9 = *(unsigned int *)(a1 + 88);
    *(_BYTE *)(a1 + 96) = 0;
    if ( (_DWORD)v9 )
    {
      v10 = *(_QWORD **)(a1 + 72);
      v11 = &v10[2 * v9];
      do
      {
        if ( *v10 != -8192 && *v10 != -4096 )
          sub_9C6650(v10 + 1);
        v10 += 2;
      }
      while ( v11 != v10 );
      v9 = *(unsigned int *)(a1 + 88);
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 72), 16 * v9, 8);
  }
  v4 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v4 )
  {
    v6 = *(_QWORD **)(a1 + 40);
    v12[0] = 2;
    v12[1] = 0;
    v12[2] = -4096;
    v7 = &v6[6 * v4];
    v12[3] = 0;
    v14[0] = 2;
    v14[1] = 0;
    v14[2] = -8192;
    v13 = &unk_49DDFA0;
    v14[3] = 0;
    do
    {
      v8 = v6[3];
      *v6 = &unk_49DB368;
      if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
        sub_BD60C0(v6 + 1);
      v6 += 6;
    }
    while ( v7 != v6 );
    v13 = &unk_49DB368;
    sub_D68D70(v14);
    sub_D68D70(v12);
    v4 = *(unsigned int *)(a1 + 56);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 40), 48 * v4, 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 24LL * *(unsigned int *)(a1 + 24), 8);
}

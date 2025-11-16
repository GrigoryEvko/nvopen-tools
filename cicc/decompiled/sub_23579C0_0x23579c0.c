// Function: sub_23579C0
// Address: 0x23579c0
//
void __fastcall sub_23579C0(unsigned __int64 *a1, __int64 *a2)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // r13
  __int64 v4; // rax
  int v5; // edx
  char v6; // bl
  __int64 v7; // r14
  __int64 v8; // rax
  int v9; // [rsp+4h] [rbp-4Ch]
  __int64 v10; // [rsp+8h] [rbp-48h]
  unsigned __int64 v11[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = a2[3];
  v3 = a2[4];
  a2[3] = 0;
  v4 = a2[1];
  v5 = *((_DWORD *)a2 + 4);
  a2[4] = 0;
  v6 = *((_BYTE *)a2 + 20);
  v7 = *a2;
  v10 = v4;
  v9 = v5;
  v8 = sub_22077B0(0x30u);
  if ( v8 )
  {
    *(_QWORD *)(v8 + 8) = v7;
    *(_BYTE *)(v8 + 28) = v6;
    *(_QWORD *)(v8 + 16) = v10;
    *(_QWORD *)(v8 + 32) = v2;
    *(_QWORD *)v8 = &unk_4A0E178;
    *(_QWORD *)(v8 + 40) = v3;
    *(_DWORD *)(v8 + 24) = v9;
    v11[0] = v8;
    sub_2356EF0(a1, v11);
    if ( v11[0] )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
  }
  else
  {
    v11[0] = 0;
    sub_2356EF0(a1, v11);
    if ( v11[0] )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
    if ( v3 )
    {
      sub_23C6FB0(v3);
      j_j___libc_free_0(v3);
    }
    if ( v2 )
    {
      sub_23C6FB0(v2);
      j_j___libc_free_0(v2);
    }
  }
}

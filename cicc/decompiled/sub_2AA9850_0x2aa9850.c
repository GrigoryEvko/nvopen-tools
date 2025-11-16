// Function: sub_2AA9850
// Address: 0x2aa9850
//
void __fastcall sub_2AA9850(__int64 a1)
{
  __int64 *v2; // r14
  unsigned __int64 v3; // r12
  __int64 v4; // r15
  _QWORD *v5; // rdi
  __int64 v6; // rsi
  _QWORD *v7; // rax
  int v8; // r8d
  __int64 v9[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  *(_QWORD *)a1 = &unk_4A23170;
  v3 = (unsigned __int64)&v2[*(unsigned int *)(a1 + 16)];
  if ( v2 != (__int64 *)v3 )
  {
    do
    {
      v4 = *v2;
      v9[0] = a1;
      v5 = *(_QWORD **)(v4 + 16);
      v6 = (__int64)&v5[*(unsigned int *)(v4 + 24)];
      v7 = sub_2AA89B0(v5, v6, v9);
      if ( (_QWORD *)v6 != v7 )
      {
        if ( (_QWORD *)v6 != v7 + 1 )
        {
          memmove(v7, v7 + 1, v6 - (_QWORD)(v7 + 1));
          v8 = *(_DWORD *)(v4 + 24);
        }
        *(_DWORD *)(v4 + 24) = v8 - 1;
      }
      ++v2;
    }
    while ( (__int64 *)v3 != v2 );
    v3 = *(_QWORD *)(a1 + 8);
  }
  if ( v3 != a1 + 24 )
    _libc_free(v3);
}

// Function: sub_230D290
// Address: 0x230d290
//
void __fastcall sub_230D290(unsigned __int64 a1)
{
  int v2; // eax
  unsigned __int64 v3; // rdi
  __int64 v4; // r15
  __int64 v5; // r15
  __int64 v6; // rbx
  _QWORD *v7; // r12
  _QWORD *v8; // r14
  __int64 v9; // r8
  __int64 (__fastcall *v10)(_QWORD *); // rax
  unsigned __int64 v11; // rdi
  __int64 v12; // [rsp+8h] [rbp-38h]
  __int64 v13; // [rsp+8h] [rbp-38h]
  __int64 v14; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = &unk_4A0B498;
  v2 = *(_DWORD *)(a1 + 20);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)v4 )
    {
      v5 = 8 * v4;
      v6 = 0;
      do
      {
        v7 = *(_QWORD **)(v3 + v6);
        if ( v7 != (_QWORD *)-8LL && v7 )
        {
          v8 = (_QWORD *)v7[1];
          v9 = *v7 + 17LL;
          if ( v8 )
          {
            v10 = *(__int64 (__fastcall **)(_QWORD *))(*v8 + 8LL);
            if ( v10 == sub_BD9990 )
            {
              v11 = v8[1];
              *v8 = &unk_49DB390;
              if ( (_QWORD *)v11 != v8 + 3 )
              {
                v12 = v9;
                j_j___libc_free_0(v11);
                v9 = v12;
              }
              v13 = v9;
              j_j___libc_free_0((unsigned __int64)v8);
              v9 = v13;
            }
            else
            {
              v14 = *v7 + 17LL;
              v10(v8);
              v9 = v14;
            }
          }
          sub_C7D6A0((__int64)v7, v9, 8);
          v3 = *(_QWORD *)(a1 + 8);
        }
        v6 += 8;
      }
      while ( v5 != v6 );
    }
  }
  _libc_free(v3);
  j_j___libc_free_0(a1);
}

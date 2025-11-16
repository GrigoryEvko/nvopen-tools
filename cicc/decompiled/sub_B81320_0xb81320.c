// Function: sub_B81320
// Address: 0xb81320
//
void __fastcall sub_B81320(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rdi
  __int64 **v6; // rsi
  __int64 *v7; // rdi
  __int64 v8; // r13
  __int64 *v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  _WORD *v14; // rdx
  __int64 v15; // r15
  __int64 *v16; // [rsp-D0h] [rbp-D0h]
  _QWORD v17[2]; // [rsp-C8h] [rbp-C8h] BYREF
  _QWORD v18[2]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 *v19; // [rsp-A8h] [rbp-A8h] BYREF
  __int64 v20; // [rsp-A0h] [rbp-A0h]
  _BYTE v21[152]; // [rsp-98h] [rbp-98h] BYREF

  if ( (int)qword_4F81B88 > 3 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v19 = (__int64 *)v21;
    v20 = 0xC00000000LL;
    if ( v3 )
    {
      v6 = &v19;
      sub_B809B0(v3, (__int64)&v19, a2);
      v7 = v19;
      v16 = &v19[(unsigned int)v20];
      if ( v16 != v19 )
      {
        v8 = (unsigned int)(2 * a3);
        v9 = v19;
        do
        {
          v10 = *v9;
          v11 = sub_C5F790();
          v14 = *(_WORD **)(v11 + 32);
          v15 = v11;
          if ( *(_QWORD *)(v11 + 24) - (_QWORD)v14 > 1u )
          {
            *v14 = 11565;
            *(_QWORD *)(v11 + 32) += 2LL;
          }
          else
          {
            v15 = sub_CB6200(v11, "--", 2);
          }
          v17[0] = v18;
          sub_2240A50(v17, v8, 32, v12, v13);
          sub_CB6200(v15, v17[0], v17[1]);
          if ( (_QWORD *)v17[0] != v18 )
            j_j___libc_free_0(v17[0], v18[0] + 1LL);
          v6 = 0;
          ++v9;
          (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v10 + 136LL))(v10, 0);
        }
        while ( v16 != v9 );
        v7 = v19;
      }
      if ( v7 != (__int64 *)v21 )
        _libc_free(v7, v6);
    }
  }
}

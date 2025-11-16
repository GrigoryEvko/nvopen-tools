// Function: sub_160EBB0
// Address: 0x160ebb0
//
void __fastcall sub_160EBB0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rdi
  _BYTE **v6; // rsi
  __int64 v7; // rdx
  _BYTE *v8; // rdi
  __int64 v9; // r14
  _QWORD *v10; // rbx
  _BYTE *v11; // r12
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  _WORD *v15; // rdx
  __int64 v16; // r15
  _QWORD *v17; // [rsp+18h] [rbp-C8h]
  const char *v18[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v19[2]; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE *v20; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v21; // [rsp+48h] [rbp-98h]
  _BYTE v22[144]; // [rsp+50h] [rbp-90h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  v20 = v22;
  v21 = 0xC00000000LL;
  if ( v3 )
  {
    v6 = &v20;
    sub_160E770(v3, (__int64)&v20, a2);
    v8 = v20;
    v17 = &v20[8 * (unsigned int)v21];
    if ( v17 != (_QWORD *)v20 )
    {
      v9 = (unsigned int)(2 * a3);
      v10 = v20;
      do
      {
        v11 = (_BYTE *)*v10;
        v12 = sub_16BA580(v8, v6, v7);
        v15 = *(_WORD **)(v12 + 24);
        v16 = v12;
        if ( *(_QWORD *)(v12 + 16) - (_QWORD)v15 > 1u )
        {
          *v15 = 11565;
          *(_QWORD *)(v12 + 24) += 2LL;
        }
        else
        {
          v16 = sub_16E7EE0(v12, "--", 2);
        }
        v18[0] = (const char *)v19;
        sub_2240A50(v18, v9, 32, v13, v14);
        sub_16E7EE0(v16, v18[0], v18[1]);
        if ( (_QWORD *)v18[0] != v19 )
          j_j___libc_free_0(v18[0], v19[0] + 1LL);
        v6 = 0;
        v8 = v11;
        ++v10;
        (*(void (__fastcall **)(_BYTE *, _QWORD))(*(_QWORD *)v11 + 136LL))(v11, 0);
      }
      while ( v17 != v10 );
      v8 = v20;
    }
    if ( v8 != v22 )
      _libc_free((unsigned __int64)v8);
  }
}

// Function: sub_38D6740
// Address: 0x38d6740
//
void __fastcall sub_38D6740(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned int *v9; // rsi
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdi
  int v13; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v14; // [rsp+8h] [rbp-D8h]
  _BYTE *v15; // [rsp+10h] [rbp-D0h]
  __int64 v16; // [rsp+18h] [rbp-C8h]
  _BYTE v17[128]; // [rsp+20h] [rbp-C0h] BYREF
  int v18; // [rsp+A0h] [rbp-40h]

  sub_38DDB90(a1, a2, a3, 0);
  v6 = *(unsigned int *)(a1 + 120);
  if ( !(_DWORD)v6 )
  {
    MEMORY[0x2C] &= ~2u;
    BUG();
  }
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v6 - 32);
  *(_BYTE *)(v7 + 44) |= 2u;
  sub_39120A0(a1);
  v8 = *(unsigned int *)(a1 + 120);
  v9 = 0;
  if ( (_DWORD)v8 )
    v9 = *(unsigned int **)(*(_QWORD *)(a1 + 112) + 32 * v8 - 32);
  sub_38CB070((_QWORD *)a1, v9);
  v10 = *(_QWORD *)(a1 + 264);
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(v10 + 8) + 80LL))(
         *(_QWORD *)(v10 + 8),
         a2,
         a3) )
  {
    if ( (*(_BYTE *)(v10 + 484) & 1) != 0 || *(_DWORD *)(v10 + 480) && *(_DWORD *)(v7 + 36) )
    {
      v14 = 0;
      v16 = 0x800000000LL;
      v11 = *(_QWORD *)(a1 + 264);
      v15 = v17;
      v18 = 0;
      v12 = *(_QWORD *)(v11 + 8);
      v13 = 0;
      (*(void (__fastcall **)(__int64, __int64, __int64, int *))(*(_QWORD *)v12 + 104LL))(v12, a2, a3, &v13);
      while ( (*(unsigned __int8 (__fastcall **)(_QWORD, int *, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 264) + 8LL)
                                                                        + 80LL))(
                *(_QWORD *)(*(_QWORD *)(a1 + 264) + 8LL),
                &v13,
                a3) )
        (*(void (__fastcall **)(_QWORD, int *, __int64, int *))(**(_QWORD **)(*(_QWORD *)(a1 + 264) + 8LL) + 104LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 264) + 8LL),
          &v13,
          a3,
          &v13);
      (*(void (__fastcall **)(__int64, int *, __int64))(*(_QWORD *)a1 + 1064LL))(a1, &v13, a3);
      if ( v15 != v17 )
        _libc_free((unsigned __int64)v15);
    }
    else
    {
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 1080LL))(a1, a2, a3);
    }
  }
  else
  {
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a1 + 1064LL))(a1, a2, a3);
  }
}

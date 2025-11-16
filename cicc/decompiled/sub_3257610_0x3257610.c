// Function: sub_3257610
// Address: 0x3257610
//
void __fastcall sub_3257610(__int64 a1, __int64 a2)
{
  __int64 *v3; // rcx
  __int64 *v4; // rdi
  unsigned __int64 v5; // r8
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // r10
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r13
  _QWORD *v18; // r15
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  int v22; // r11d
  const char *v23; // [rsp+0h] [rbp-60h] BYREF
  char v24; // [rsp+20h] [rbp-40h]
  char v25; // [rsp+21h] [rbp-3Fh]

  v3 = *(__int64 **)(a2 + 432);
  v4 = *(__int64 **)(a2 + 440);
  if ( v3 != v4 )
  {
    v5 = *(_QWORD *)(a2 + 496);
    v6 = *(unsigned int *)(a2 + 512);
    v7 = (unsigned int)(v6 - 1);
    do
    {
      v8 = *v3;
      if ( (_DWORD)v6 )
      {
        v9 = v7 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v10 = *(_QWORD *)(v5 + 16LL * v9);
        if ( v8 == v10 )
        {
LABEL_5:
          v11 = sub_32530C0((char *)a1, v6, v10, (__int64)v3, v5, v7);
          v12 = *(_QWORD *)(a1 + 8);
          v25 = 1;
          v13 = v11;
          v24 = 3;
          v23 = "GCC_except_table_end";
          v17 = sub_31DCC50(v12, (__int64 *)&v23, v14, v15, v16);
          (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
            *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
            v17,
            0);
          v18 = *(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL) + 8LL);
          v19 = sub_E808D0(v13, 0, v18, 0);
          v20 = sub_E808D0(v17, 0, v18, 0);
          v21 = sub_E81A00(18, v20, v19, v18, 0);
          (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 448LL))(
            *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
            v13,
            v21);
          return;
        }
        v22 = 1;
        while ( v10 != -4096 )
        {
          v9 = v7 & (v22 + v9);
          v10 = *(_QWORD *)(v5 + 16LL * v9);
          if ( v8 == v10 )
            goto LABEL_5;
          ++v22;
        }
      }
      v3 += 15;
    }
    while ( v4 != v3 );
  }
}

// Function: sub_2F28510
// Address: 0x2f28510
//
__int64 __fastcall sub_2F28510(_QWORD *a1, __int64 a2)
{
  void *v4; // rsi
  unsigned int v5; // eax
  __int64 v6; // r8
  __int64 v7; // r9
  _DWORD *v8; // rdi
  unsigned int v9; // r12d
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 (*v13)(); // rdx
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r13
  int v17; // eax
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // [rsp+0h] [rbp-40h] BYREF
  __int64 i; // [rsp+8h] [rbp-38h]
  char v25; // [rsp+10h] [rbp-30h]
  __int64 v26; // [rsp+18h] [rbp-28h]

  v4 = *(void **)a2;
  v5 = sub_BB98D0(a1, (__int64)v4);
  if ( (_BYTE)v5 )
    return 0;
  v8 = *(_DWORD **)(a2 + 16);
  v9 = v5;
  v10 = 0;
  v11 = 0;
  if ( (unsigned int)(v8[10] - 42) <= 1 )
  {
    v19 = (__int64 *)a1[1];
    v20 = *v19;
    v21 = v19[1];
    if ( v20 == v21 )
LABEL_14:
      BUG();
    v4 = &unk_501FE44;
    while ( *(_UNKNOWN **)v20 != &unk_501FE44 )
    {
      v20 += 16;
      if ( v21 == v20 )
        goto LABEL_14;
    }
    v22 = (*(__int64 (__fastcall **)(_QWORD, void *, __int64, _QWORD))(**(_QWORD **)(v20 + 8) + 104LL))(
            *(_QWORD *)(v20 + 8),
            &unk_501FE44,
            v21,
            0);
    v8 = *(_DWORD **)(a2 + 16);
    v10 = 1;
    v11 = v22 + 200;
  }
  v12 = *(_QWORD *)(a2 + 32);
  v26 = v11;
  i = 0;
  v25 = v10;
  v23 = v12;
  v13 = *(__int64 (**)())(*(_QWORD *)v8 + 128LL);
  v14 = 0;
  if ( v13 != sub_2DAC790 )
    v14 = ((__int64 (__fastcall *)(_DWORD *, void *, __int64 (*)(), __int64))v13)(v8, v4, v13, v10);
  v15 = *(_QWORD *)(a2 + 328);
  v16 = a2 + 320;
  for ( i = v14; v16 != v15; v9 |= v17 )
  {
    v17 = sub_2F26B60(&v23, v15, (__int64)v13, v10, v6, v7);
    v15 = *(_QWORD *)(v15 + 8);
  }
  return v9;
}

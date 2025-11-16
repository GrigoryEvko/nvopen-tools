// Function: sub_2F796A0
// Address: 0x2f796a0
//
void __fastcall sub_2F796A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7, char a8)
{
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rax
  unsigned int v15; // eax
  unsigned int v16; // r12d
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  int v22[13]; // [rsp+2Ch] [rbp-34h] BYREF

  sub_2F75310(a1);
  *(_QWORD *)a1 = a2;
  v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  *(_QWORD *)(a1 + 8) = v10;
  v11 = v10;
  v12 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 16) = a3;
  v13 = *(_BYTE *)(a1 + 56) == 0;
  v14 = *(_QWORD *)(v12 + 32);
  *(_QWORD *)(a1 + 40) = a5;
  *(_BYTE *)(a1 + 57) = a8;
  *(_QWORD *)(a1 + 24) = v14;
  *(_BYTE *)(a1 + 58) = a7;
  if ( !v13 )
    *(_QWORD *)(a1 + 32) = a4;
  *(_QWORD *)(a1 + 64) = a6;
  v22[0] = 0;
  v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 392LL))(v11);
  sub_1D05C60(a1 + 72, v15, v22);
  sub_2F74200(*(_QWORD *)(a1 + 48), (char **)(a1 + 72));
  sub_2F75250(a1 + 96, *(_DWORD **)(a1 + 24));
  if ( a8 )
  {
    v16 = *(_DWORD *)(*(_QWORD *)(a1 + 24) + 64LL);
    if ( v16 < *(_DWORD *)(a1 + 384) >> 2 || v16 > *(_DWORD *)(a1 + 384) )
    {
      v17 = (__int64)_libc_calloc(v16, 1u);
      if ( !v17 && (v16 || (v17 = malloc(1u)) == 0) )
        sub_C64F00("Allocation failed", 1u);
      v18 = *(_QWORD *)(a1 + 376);
      *(_QWORD *)(a1 + 376) = v17;
      if ( v18 )
        _libc_free(v18);
      *(_DWORD *)(a1 + 384) = v16;
    }
  }
}

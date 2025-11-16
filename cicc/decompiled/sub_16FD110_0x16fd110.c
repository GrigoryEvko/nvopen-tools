// Function: sub_16FD110
// Address: 0x16fd110
//
__int64 __fastcall sub_16FD110(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbp
  __int64 result; // rax
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 *v15; // rax
  __int64 v16; // r9
  _QWORD *v17; // [rsp-60h] [rbp-60h]
  __int64 v18; // [rsp-58h] [rbp-58h] BYREF
  _QWORD *v19; // [rsp-40h] [rbp-40h]
  _QWORD v20[6]; // [rsp-30h] [rbp-30h] BYREF

  result = *(_QWORD *)(a1 + 72);
  if ( !result )
  {
    v20[5] = v5;
    v7 = *(_DWORD *)sub_16FC340(a1, a2, a3, a4, a5);
    if ( (v7 & 0xFFFFFFF7) == 0 || v7 == 17 )
      goto LABEL_8;
    if ( v7 == 16 )
    {
      a2 = a1;
      sub_16FC240((__int64)&v18, a1, v8, v9, v10);
      if ( v19 != v20 )
      {
        a2 = v20[0] + 1LL;
        j_j___libc_free_0(v19, v20[0] + 1LL);
      }
    }
    v11 = *(_DWORD *)sub_16FC340(a1, a2, v8, v9, v10);
    if ( v11 == 8 || v11 == 17 )
    {
LABEL_8:
      v15 = (__int64 *)sub_16F82D0(a1);
      v17 = (_QWORD *)sub_145CBF0(v15, 72, 16);
      sub_16FC350((__int64)v17, 0, *(_QWORD *)(a1 + 8), 0, 0, v16, 0);
      *v17 = &unk_49EFDB8;
      *(_QWORD *)(a1 + 72) = v17;
      return (__int64)v17;
    }
    else
    {
      result = sub_16FD100(a1, a2, v12, v13, v14);
      *(_QWORD *)(a1 + 72) = result;
    }
  }
  return result;
}

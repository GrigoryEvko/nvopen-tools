// Function: sub_17CFEC0
// Address: 0x17cfec0
//
unsigned __int16 __fastcall sub_17CFEC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int16 result; // ax
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 *v10; // rax
  _QWORD *v11; // r12
  __int64 **v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 *v17; // rax
  __int64 v18[3]; // [rsp-78h] [rbp-78h] BYREF
  _QWORD *v19; // [rsp-60h] [rbp-60h]

  result = (*(_WORD *)(*(_QWORD *)(a1 + 8) + 18LL) >> 4) & 0x3FF;
  if ( result != 79 )
  {
    v7 = *(unsigned int *)(a1 + 56);
    if ( (unsigned int)v7 >= *(_DWORD *)(a1 + 60) )
    {
      sub_16CD150(a1 + 48, (const void *)(a1 + 64), 0, 8, a5, a6);
      v7 = *(unsigned int *)(a1 + 56);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v7) = a2;
    ++*(_DWORD *)(a1 + 56);
    sub_17CE510((__int64)v18, a2, 0, 0, 0);
    v8 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v9 = *(_QWORD *)(a1 + 24);
    v10 = (__int64 *)sub_1643330(v19);
    v11 = (_QWORD *)sub_17CFB40(v9, v8, v18, v10, 8u);
    v12 = (__int64 **)sub_1643330(v19);
    v15 = sub_15A06D0(v12, v8, v13, v14);
    v16 = sub_1643360(v19);
    v17 = (__int64 *)sub_159C470(v16, 24, 0);
    result = (unsigned __int16)sub_15E7280(v18, v11, v15, v17, 8u, 0, 0, 0, 0);
    if ( v18[0] )
      return sub_161E7C0((__int64)v18, v18[0]);
  }
  return result;
}

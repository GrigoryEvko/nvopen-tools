// Function: sub_E8CCE0
// Address: 0xe8cce0
//
__int64 __fastcall sub_E8CCE0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int64 v7; // rax
  _QWORD *v8; // rdi
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 v14; // rax
  unsigned __int16 v16; // [rsp+Dh] [rbp-33h]
  unsigned __int8 v17; // [rsp+Fh] [rbp-31h]

  if ( a3 )
  {
    v7 = sub_E8A230((__int64)a1, a4, a3, 0);
    v8 = (_QWORD *)a1[1];
    v9 = v7;
    v10 = v8[36];
    v8[46] += 128LL;
    v11 = (v10 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8[37] >= (unsigned __int64)(v11 + 128) && v10 )
      v8[36] = v11 + 128;
    else
      v11 = sub_9D1E70((__int64)(v8 + 36), 128, 128, 3);
    sub_E81B30(v11, 6, 0);
    *(_BYTE *)(v11 + 30) = 0;
    *(_QWORD *)(v11 + 40) = v11 + 64;
    *(_QWORD *)(v11 + 72) = v11 + 88;
    *(_QWORD *)(v11 + 32) = 0;
    *(_QWORD *)(v11 + 48) = 0;
    *(_QWORD *)(v11 + 56) = 8;
    *(_QWORD *)(v11 + 80) = 0x100000000LL;
    *(_QWORD *)(v11 + 112) = a2;
    *(_QWORD *)(v11 + 120) = v9;
    v12 = *(_QWORD *)(a1[36] + 8LL);
    *(_QWORD *)(v11 + 8) = v12;
    *(_DWORD *)(v11 + 24) = *(_DWORD *)(a1[36] + 24LL) + 1;
    *(_QWORD *)a1[36] = v11;
    a1[36] = v11;
    result = *(_QWORD *)(v12 + 8);
    *(_QWORD *)(result + 8) = v11;
  }
  else
  {
    v14 = a1[37];
    v16 = *(_WORD *)(v14 + 72);
    v17 = *(_BYTE *)(v14 + 74);
    (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a1 + 536LL))(a1, 0, 1);
    sub_E98EB0(a1, (int)(a5 + 1), 0);
    (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*a1 + 536LL))(a1, 2, 1);
    sub_E9A500(a1, a4, a5, 0);
    return sub_E77F70(a1, v16 | (v17 << 16), a2, 0);
  }
  return result;
}

// Function: sub_305B3C0
// Address: 0x305b3c0
//
_QWORD *__fastcall sub_305B3C0(__int64 a1, int a2, __int64 a3, __int64 *a4, __int64 a5)
{
  _DWORD *v9; // rax
  _QWORD *result; // rax
  void *v11; // rdx
  __int64 v12; // rdi
  void *v13; // [rsp-10h] [rbp-30h]

  sub_305AFE0((_QWORD *)a1, a2, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), *(_QWORD *)a3, *(_QWORD *)(a3 + 8), *a4, a4[1]);
  *(_QWORD *)(a1 + 312) = 0;
  *(_BYTE *)(a1 + 320) = 0;
  *(_BYTE *)(a1 + 368) = 0;
  *(_QWORD *)a1 = &unk_4A305E8;
  *(_QWORD *)(a1 + 304) = a1 + 320;
  *(_QWORD *)(a1 + 336) = 0x20800000000LL;
  *(_WORD *)(a1 + 348) = 0;
  *(_DWORD *)(a1 + 344) = 52;
  *(_DWORD *)(a1 + 364) = 0;
  sub_36D65B0(a1 + 376);
  v9 = (_DWORD *)sub_305B070(a1, *(char **)a3, *(_QWORD *)(a3 + 8), (_BYTE *)*a4, a4[1]);
  sub_3052FA0(a1 + 960, a5, v9);
  *(_QWORD *)(a1 + 537992) = 0;
  sub_36D49D0(a1 + 538000);
  result = (_QWORD *)sub_22077B0(8u);
  v11 = v13;
  if ( result )
  {
    v11 = &unk_4A3C328;
    *result = &unk_4A3C328;
  }
  v12 = *(_QWORD *)(a1 + 537992);
  *(_QWORD *)(a1 + 537992) = result;
  if ( v12 )
    return (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, void *))(*(_QWORD *)v12 + 8LL))(v12, a5, v11);
  return result;
}

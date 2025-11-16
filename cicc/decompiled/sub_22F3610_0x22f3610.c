// Function: sub_22F3610
// Address: 0x22f3610
//
__int64 __fastcall sub_22F3610(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  size_t v8; // r15
  _QWORD *v10; // rdi
  unsigned __int64 v11; // r13
  __int64 result; // rax

  v6 = a1 + 24;
  v8 = a3 - a2;
  v10 = (_QWORD *)(a1 + 200);
  v11 = (a3 - a2) >> 3;
  *(v10 - 24) = v6;
  *(v10 - 23) = 0x1000000000LL;
  *(v10 - 6) = 0;
  *(v10 - 5) = 0;
  *(v10 - 4) = 0;
  *((_DWORD *)v10 - 6) = 0;
  *(v10 - 25) = &unk_4A0AA50;
  *(_QWORD *)(a1 + 192) = 0x1000000000LL;
  *(_QWORD *)(a1 + 336) = a1 + 328;
  *(_QWORD *)(a1 + 328) = a1 + 328;
  result = 0;
  *(_QWORD *)(a1 + 184) = v10;
  *(_QWORD *)(a1 + 344) = 0;
  *(_DWORD *)(a1 + 352) = v11;
  if ( (unsigned __int64)(a3 - a2) > 0x80 )
  {
    sub_C8D5F0(a1 + 184, v10, v11, 8u, a1 + 184, a6);
    result = *(unsigned int *)(a1 + 192);
    v10 = (_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * result);
  }
  if ( a3 != a2 )
  {
    memcpy(v10, a2, v8);
    result = *(unsigned int *)(a1 + 192);
  }
  *(_DWORD *)(a1 + 192) = result + v11;
  return result;
}

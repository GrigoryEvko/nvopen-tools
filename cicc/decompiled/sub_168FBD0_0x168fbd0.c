// Function: sub_168FBD0
// Address: 0x168fbd0
//
__int64 __fastcall sub_168FBD0(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v3; // rax
  size_t v5; // r15
  _QWORD *v7; // rdi
  __int64 v8; // r13
  __int64 result; // rax

  v3 = a1 + 24;
  v5 = a3 - a2;
  v7 = (_QWORD *)(a1 + 200);
  v8 = (a3 - a2) >> 3;
  *(v7 - 24) = v3;
  *(v7 - 23) = 0x1000000000LL;
  *(v7 - 6) = 0;
  *(v7 - 5) = 0;
  *(v7 - 4) = 0;
  *((_DWORD *)v7 - 6) = 0;
  *(v7 - 25) = &unk_49EE9E8;
  *(_QWORD *)(a1 + 192) = 0x1000000000LL;
  *(_QWORD *)(a1 + 336) = a1 + 328;
  *(_QWORD *)(a1 + 328) = a1 + 328;
  result = 0;
  *(_QWORD *)(a1 + 184) = v7;
  *(_QWORD *)(a1 + 344) = 0;
  *(_DWORD *)(a1 + 352) = v8;
  if ( (unsigned __int64)(a3 - a2) > 0x80 )
  {
    sub_16CD150(a1 + 184, v7, v8, 8);
    result = *(unsigned int *)(a1 + 192);
    v7 = (_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * result);
  }
  if ( a3 != a2 )
  {
    memcpy(v7, a2, v5);
    result = *(unsigned int *)(a1 + 192);
  }
  *(_DWORD *)(a1 + 192) = result + v8;
  return result;
}

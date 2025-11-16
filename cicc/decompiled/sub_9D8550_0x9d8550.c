// Function: sub_9D8550
// Address: 0x9d8550
//
void __fastcall sub_9D8550(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v4; // rax
  _QWORD *v5; // rax
  bool v6; // dl
  __int64 *v7; // rbx
  char *v8; // rsi

  v4 = *a3;
  if ( *(_DWORD *)(*a3 + 8LL) == 1 )
  {
    v5 = *(_QWORD **)(v4 + 88);
    v6 = 0;
    if ( v5 )
      v6 = *v5 != v5[1];
    *(_BYTE *)(a1 + 347) |= v6;
    v4 = *a3;
  }
  v7 = (__int64 *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  sub_9D82B0(a1, *(_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL), *(_QWORD *)(v4 + 16));
  v8 = *(char **)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 32);
  if ( v8 == (char *)v7[5] )
  {
    sub_9D0210(v7 + 3, v8, a3);
  }
  else
  {
    if ( v8 )
    {
      *(_QWORD *)v8 = *a3;
      *a3 = 0;
      v8 = (char *)v7[4];
    }
    v7[4] = (__int64)(v8 + 8);
  }
}

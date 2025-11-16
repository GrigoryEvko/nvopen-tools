// Function: sub_C7DD80
// Address: 0xc7dd80
//
__int64 __fastcall sub_C7DD80(__int64 a1, const void *a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v14; // rax
  _QWORD v15[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_C7DB40(v15, a3, a4, 0, a5, a6);
  v12 = v15[0];
  if ( v15[0] )
  {
    if ( a3 )
    {
      memmove(*(void **)(v15[0] + 8LL), a2, a3);
      v12 = v15[0];
    }
    *(_QWORD *)a1 = v12;
    *(_BYTE *)(a1 + 16) &= ~1u;
    return a1;
  }
  else
  {
    v14 = sub_2241E50(v15, a3, v9, v10, v11);
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_QWORD *)(a1 + 8) = v14;
    *(_DWORD *)a1 = 12;
    return a1;
  }
}

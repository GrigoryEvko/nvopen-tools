// Function: sub_16C26E0
// Address: 0x16c26e0
//
__int64 __fastcall sub_16C26E0(__int64 a1, const void *a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v12; // rax
  _QWORD v13[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_16C2500(v13, a3, a4);
  if ( v13[0] )
  {
    memcpy(*(void **)(v13[0] + 8LL), a2, a3);
    v10 = v13[0];
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v10;
  }
  else
  {
    v12 = sub_2241E50(v13, a3, v7, v8, v9);
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_QWORD *)(a1 + 8) = v12;
    *(_DWORD *)a1 = 12;
  }
  return a1;
}

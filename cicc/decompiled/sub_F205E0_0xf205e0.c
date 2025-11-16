// Function: sub_F205E0
// Address: 0xf205e0
//
void __fastcall sub_F205E0(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 *v7; // r9
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10[3]; // [rsp+8h] [rbp-18h] BYREF

  v4 = *a1 + 2096LL;
  v10[0] = *a2;
  v3 = v10[0];
  sub_F200C0(v4, v10);
  if ( *(_BYTE *)v3 == 85 )
  {
    v8 = *(_QWORD *)(v3 - 32);
    if ( v8 )
    {
      if ( !*(_BYTE *)v8 )
      {
        v9 = *(_QWORD *)(v3 + 80);
        if ( *(_QWORD *)(v8 + 24) == v9 && (*(_BYTE *)(v8 + 33) & 0x20) != 0 && *(_DWORD *)(v8 + 36) == 11 )
          sub_CFEAE0(a1[1], v3, v9, v5, v6, v7);
      }
    }
  }
}

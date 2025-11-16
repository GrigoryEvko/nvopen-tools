// Function: sub_38C5570
// Address: 0x38c5570
//
void __fastcall sub_38C5570(__int64 a1, _QWORD *a2)
{
  int v3; // r8d
  int v4; // r9d
  unsigned __int64 v5; // rbx
  _BYTE *v6; // rsi
  _BYTE *v7; // rdx
  _BYTE *v8; // rax
  _BYTE *v9; // [rsp+0h] [rbp-30h] BYREF
  __int64 v10; // [rsp+8h] [rbp-28h]
  _BYTE v11[32]; // [rsp+10h] [rbp-20h] BYREF

  (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a2 + 160LL))(a2, *(_QWORD *)(*(_QWORD *)(a2[1] + 32LL) + 96LL), 0);
  sub_16805A0(a1 + 8);
  v5 = *(_QWORD *)(a1 + 40);
  v10 = 0;
  v9 = v11;
  v6 = v11;
  if ( v5 )
  {
    sub_16CD150((__int64)&v9, v11, v5, 1, v3, v4);
    v6 = v9;
    v7 = &v9[v5];
    v8 = &v9[(unsigned int)v10];
    if ( v8 != &v9[v5] )
    {
      do
      {
        if ( v8 )
          *v8 = 0;
        ++v8;
      }
      while ( v7 != v8 );
      v6 = v9;
    }
    LODWORD(v10) = v5;
  }
  sub_167FAF0(a1 + 8, v6);
  (*(void (__fastcall **)(_QWORD *, _BYTE *, _QWORD))(*a2 + 408LL))(a2, v9, (unsigned int)v10);
  if ( v9 != v11 )
    _libc_free((unsigned __int64)v9);
}

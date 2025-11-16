// Function: sub_3706C40
// Address: 0x3706c40
//
__int64 __fastcall sub_3706C40(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  void (*v5)(void); // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  _QWORD *v10; // r13
  unsigned __int64 v11; // rdi
  volatile signed __int32 *v12; // rdi
  __int64 v14; // [rsp+8h] [rbp-38h]

  v5 = *(void (**)(void))(*a2 + 24LL);
  if ( (char *)v5 == (char *)sub_3706B00 )
  {
    v6 = a3[1] - 4LL;
    v14 = *a3 + 4LL;
    v7 = sub_22077B0(0xB8u);
    v8 = v7;
    if ( v7 )
    {
      *(_QWORD *)(v7 + 24) = v6;
      v9 = v7 + 32;
      *(_DWORD *)(v7 + 8) = 1;
      *(_QWORD *)v7 = &unk_49E6828;
      *(_QWORD *)(v7 + 16) = v14;
      sub_12548A0((_QWORD *)(v7 + 32));
      *(_BYTE *)(v8 + 106) = 0;
      *(_BYTE *)(v8 + 110) = 0;
      *(_QWORD *)(v8 + 152) = v9;
      *(_QWORD *)(v8 + 96) = &unk_4A3C998;
      *(_QWORD *)(v8 + 112) = v8 + 128;
      *(_QWORD *)(v8 + 120) = 0x200000000LL;
      *(_QWORD *)(v8 + 160) = 0;
      *(_QWORD *)(v8 + 168) = 0;
      *(_QWORD *)(v8 + 176) = 0;
    }
    v10 = (_QWORD *)a2[1];
    a2[1] = v8;
    if ( v10 )
    {
      v11 = v10[14];
      if ( (_QWORD *)v11 != v10 + 16 )
        _libc_free(v11);
      v12 = (volatile signed __int32 *)v10[6];
      v10[4] = &unk_49E6870;
      if ( v12 )
        sub_A191D0(v12);
      j_j___libc_free_0((unsigned __int64)v10);
      v8 = a2[1];
    }
    sub_370EAB0(a1, v8 + 96, a3);
  }
  else
  {
    v5();
  }
  return a1;
}

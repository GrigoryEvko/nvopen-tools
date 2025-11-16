// Function: sub_3706B00
// Address: 0x3706b00
//
__int64 __fastcall sub_3706B00(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r14
  _QWORD *v9; // r14
  unsigned __int64 v10; // rdi
  volatile signed __int32 *v11; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v5 = a3[1] - 4LL;
  v13 = *a3 + 4LL;
  v6 = sub_22077B0(0xB8u);
  v7 = v6;
  if ( v6 )
  {
    *(_QWORD *)(v6 + 24) = v5;
    v8 = v6 + 32;
    *(_DWORD *)(v6 + 8) = 1;
    *(_QWORD *)v6 = &unk_49E6828;
    *(_QWORD *)(v6 + 16) = v13;
    sub_12548A0((_QWORD *)(v6 + 32));
    *(_BYTE *)(v7 + 106) = 0;
    *(_BYTE *)(v7 + 110) = 0;
    *(_QWORD *)(v7 + 152) = v8;
    *(_QWORD *)(v7 + 96) = &unk_4A3C998;
    *(_QWORD *)(v7 + 112) = v7 + 128;
    *(_QWORD *)(v7 + 120) = 0x200000000LL;
    *(_QWORD *)(v7 + 160) = 0;
    *(_QWORD *)(v7 + 168) = 0;
    *(_QWORD *)(v7 + 176) = 0;
  }
  v9 = *(_QWORD **)(a2 + 8);
  *(_QWORD *)(a2 + 8) = v7;
  if ( v9 )
  {
    v10 = v9[14];
    if ( (_QWORD *)v10 != v9 + 16 )
      _libc_free(v10);
    v11 = (volatile signed __int32 *)v9[6];
    v9[4] = &unk_49E6870;
    if ( v11 )
      sub_A191D0(v11);
    j_j___libc_free_0((unsigned __int64)v9);
    v7 = *(_QWORD *)(a2 + 8);
  }
  sub_370EAB0(a1, v7 + 96, a3);
  return a1;
}

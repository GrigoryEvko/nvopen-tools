// Function: sub_38D5C00
// Address: 0x38d5c00
//
void __fastcall sub_38D5C00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 (__fastcall *v4)(__int64); // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_38D3D10(a1, a3, a2);
  v4 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 72LL);
  if ( v4 == sub_38D3BD0 )
  {
    LODWORD(v5) = 0;
    if ( *(_BYTE *)(a1 + 260) )
      v5 = *(_QWORD *)(a1 + 264);
  }
  else
  {
    LODWORD(v5) = v4(a1);
  }
  if ( sub_38CF2B0(v3, v14, v5) )
  {
    sub_38C6E60((_QWORD *)a1, v14[0]);
  }
  else
  {
    v6 = sub_22077B0(0x50u);
    v7 = v6;
    if ( v6 )
    {
      v8 = v6;
      sub_38CF760(v6, 7, 0, 0);
      *(_QWORD *)(v7 + 48) = v3;
      *(_QWORD *)(v7 + 56) = v7 + 72;
      *(_QWORD *)(v7 + 64) = 0x800000001LL;
      *(_BYTE *)(v7 + 72) = 0;
    }
    else
    {
      v8 = 0;
    }
    sub_38D4150(a1, v7, 0);
    v9 = *(unsigned int *)(a1 + 120);
    v10 = 0;
    if ( (_DWORD)v9 )
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v9 - 32);
    v11 = *(__int64 **)(a1 + 272);
    v12 = *v11;
    v13 = *(_QWORD *)v7 & 7LL;
    *(_QWORD *)(v7 + 8) = v11;
    v12 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v7 = v12 | v13;
    *(_QWORD *)(v12 + 8) = v8;
    *v11 = *v11 & 7 | v8;
    *(_QWORD *)(v7 + 24) = v10;
  }
}

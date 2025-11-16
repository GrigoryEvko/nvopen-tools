// Function: sub_305FB20
// Address: 0x305fb20
//
unsigned __int64 __fastcall sub_305FB20(__int64 a1, char *a2, __int64 *a3, __int64 *a4)
{
  int v6; // edx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 *v9; // r15
  __int64 v10; // rax
  size_t v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rdx
  bool v17; // zf
  __int64 v18; // rbx
  __int64 *v19; // rax
  _QWORD v21[4]; // [rsp+10h] [rbp-60h] BYREF
  char v22; // [rsp+30h] [rbp-40h]
  char v23; // [rsp+31h] [rbp-3Fh]

  *(_QWORD *)a1 = &unk_49DC150;
  v6 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a1 + 12) & 0x8000 | 0x20;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 128;
  *(_QWORD *)(a1 + 112) = 1;
  *(_DWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 124) = 1;
  v9 = sub_C57470();
  v10 = *(unsigned int *)(a1 + 80);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v10 + 1, 8u, v7, v8);
    v10 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v10) = v9;
  ++*(_DWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)a1 = &unk_49DC380;
  v11 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v11);
  v15 = a3[1];
  v16 = *a3;
  v17 = *(_QWORD *)(a1 + 136) == 0;
  v18 = *a4;
  *(_QWORD *)(a1 + 40) = v16;
  *(_QWORD *)(a1 + 48) = v15;
  if ( !v17 )
  {
    v19 = sub_CEADF0();
    a2 = (char *)v21;
    v23 = 1;
    v21[0] = "cl::alias must only have one cl::aliasopt(...) specified!";
    v22 = 3;
    sub_C53280(a1, (__int64)v21, 0, 0, (__int64)v19);
  }
  *(_QWORD *)(a1 + 136) = v18;
  return sub_C53EE0(a1, a2, v16, v12, v13, v14);
}

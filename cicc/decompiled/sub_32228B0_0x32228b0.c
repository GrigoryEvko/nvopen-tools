// Function: sub_32228B0
// Address: 0x32228b0
//
void __fastcall sub_32228B0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r14
  __int64 v4; // rax
  char v5; // r14
  __int64 v6; // rdi
  void (__fastcall *v7)(__int64, __int64, _QWORD); // r15
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rdi
  void (__fastcall *v13)(__int64, _QWORD, _QWORD); // r15
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-40h]
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+0h] [rbp-40h]
  __int64 i; // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 656);
  v2 = 16LL * *(unsigned int *)(a1 + 664);
  for ( i = v1 + v2; i != v1; v1 += 16 )
  {
    v11 = *(_QWORD *)(v1 + 8);
    if ( (unsigned __int8)sub_37365C0(v11) )
    {
      v12 = *(_QWORD *)(a1 + 8);
      v18 = *(_QWORD *)(v12 + 224);
      v13 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v18 + 176LL);
      if ( *(_DWORD *)(*(_QWORD *)(v11 + 80) + 36LL) == 1 )
      {
        v14 = sub_31DA6B0(v12);
        v13(v18, *(_QWORD *)(v14 + 368), 0);
        v5 = 1;
        sub_3221E10(a1, 1, (__int64)"Names", 5, v11, (__int64 **)(v11 + 424));
        v15 = *(_QWORD *)(a1 + 8);
        v19 = *(_QWORD *)(v15 + 224);
        v7 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v19 + 176LL);
        v16 = sub_31DA6B0(v15);
        v9 = v19;
        v10 = *(_QWORD *)(v16 + 376);
      }
      else
      {
        v4 = sub_31DA6B0(v12);
        v13(v18, *(_QWORD *)(v4 + 184), 0);
        v5 = 0;
        sub_3221E10(a1, 0, (__int64)"Names", 5, v11, (__int64 **)(v11 + 424));
        v6 = *(_QWORD *)(a1 + 8);
        v17 = *(_QWORD *)(v6 + 224);
        v7 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v17 + 176LL);
        v8 = sub_31DA6B0(v6);
        v9 = v17;
        v10 = *(_QWORD *)(v8 + 120);
      }
      v7(v9, v10, 0);
      sub_3221E10(a1, v5, (__int64)"Types", 5, v11, (__int64 **)(v11 + 448));
    }
  }
}

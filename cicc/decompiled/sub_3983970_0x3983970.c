// Function: sub_3983970
// Address: 0x3983970
//
unsigned __int64 __fastcall sub_3983970(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void (__fastcall *(__fastcall *v6)(__int64))(__int64, __int64); // rdx
  void (*v7)(void); // rax
  unsigned __int64 v8; // rdi
  unsigned __int64 result; // rax

  v6 = sub_39837E0;
  v7 = *(void (**)(void))(*(_QWORD *)a1 + 64LL);
  if ( (char *)v7 == (char *)sub_39837E0 )
  {
    if ( *(_BYTE *)(a1 + 24) )
      sub_38DD230(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL));
  }
  else
  {
    v7();
  }
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 264LL);
  result = *(_QWORD *)(v8 + 416);
  if ( *(_QWORD *)(v8 + 408) != result )
    return sub_1E0EC90(v8, 0, (__int64)v6, a4, a5, a6);
  return result;
}

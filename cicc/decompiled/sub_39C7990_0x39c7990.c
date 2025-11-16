// Function: sub_39C7990
// Address: 0x39c7990
//
__int64 __fastcall sub_39C7990(__int64 a1, int a2, unsigned __int8 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  __int64 result; // rax
  const char *v10; // [rsp+0h] [rbp-40h] BYREF
  char v11; // [rsp+10h] [rbp-30h]
  char v12; // [rsp+11h] [rbp-2Fh]

  sub_39A1E90(a1, 17, (__int64)a3, a4, a5, a6);
  *(_DWORD *)(a1 + 600) = a2;
  *(_QWORD *)(a1 + 608) = 0;
  *(_QWORD *)(a1 + 616) = 0;
  *(_QWORD *)a1 = &unk_4A40688;
  *(_QWORD *)(a1 + 688) = 0x1000000000LL;
  *(_QWORD *)(a1 + 720) = 0x1000000000LL;
  *(_QWORD *)(a1 + 736) = a1 + 752;
  *(_QWORD *)(a1 + 744) = 0x100000000LL;
  *(_QWORD *)(a1 + 808) = a1 + 824;
  *(_QWORD *)(a1 + 816) = 0x200000000LL;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_DWORD *)(a1 + 664) = 0;
  *(_QWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 704) = 0;
  *(_QWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 856) = 0;
  *(_QWORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_DWORD *)(a1 + 888) = 0;
  *(_QWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 912) = 0;
  *(_DWORD *)(a1 + 920) = 0;
  *(_QWORD *)(a1 + 928) = 0;
  sub_39A55B0(a1, a3, (unsigned __int8 *)(a1 + 8));
  v8 = *(_QWORD *)(a1 + 192);
  v12 = 1;
  v10 = "cu_macro_begin";
  v11 = 3;
  result = sub_396F530(v8, (__int64)&v10);
  *(_QWORD *)(a1 + 632) = result;
  return result;
}

// Function: sub_11CCA60
// Address: 0x11cca60
//
__int64 __fastcall sub_11CCA60(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        __int64 *a7)
{
  __int64 v10; // rax
  char *v11; // rax
  __int64 v12; // rdx
  unsigned int v15[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v10 = sub_AA4B30(*(_QWORD *)(a6 + 48));
  v11 = sub_11C9DB0(v10, a2, *(_QWORD *)(a1 + 8), a3, a4, a5, v15);
  return sub_11CC8D0(a1, v15[0], (__int64)v11, v12, a6, a7, a2);
}

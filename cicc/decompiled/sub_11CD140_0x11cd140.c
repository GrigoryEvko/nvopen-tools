// Function: sub_11CD140
// Address: 0x11cd140
//
__int64 __fastcall sub_11CD140(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        __int64 a7,
        __int64 *a8)
{
  __int64 v10; // rax
  char *v11; // rax
  unsigned __int64 v12; // rdx
  unsigned int v16[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v10 = sub_AA4B30(*(_QWORD *)(a7 + 48));
  v11 = sub_11C9DB0(v10, a3, *(_QWORD *)(a1 + 8), a4, a5, a6, v16);
  return sub_11CCF70(a1, a2, v16[0], (__int64)v11, v12, a7, a8, a3);
}

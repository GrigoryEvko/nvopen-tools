// Function: sub_8D93A0
// Address: 0x8d93a0
//
__int64 __fastcall sub_8D93A0(
        __int64 a1,
        __int64 (__fastcall *a2)(__int64, unsigned int *),
        __int64 a3,
        unsigned int a4)
{
  __int64 result; // rax
  _QWORD *v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  _BYTE v11[64]; // [rsp+0h] [rbp-100h] BYREF
  __int64 (__fastcall *v12)(); // [rsp+40h] [rbp-C0h]
  unsigned int v13; // [rsp+50h] [rbp-B0h]
  int v14; // [rsp+54h] [rbp-ACh]
  __int64 v15; // [rsp+58h] [rbp-A8h]
  __int64 (__fastcall *v16)(__int64, unsigned int *); // [rsp+A8h] [rbp-58h]
  __int64 v17; // [rsp+B0h] [rbp-50h]
  unsigned int v18; // [rsp+B8h] [rbp-48h]

  result = sub_8D8C50(*(_QWORD *)(a1 + 128), a2, a3, a4);
  if ( (_DWORD)result )
    return 1;
  if ( *(_BYTE *)(a1 + 173) == 12 && *(_BYTE *)(a1 + 176) == 1 )
  {
    v7 = *(_QWORD **)(a1 + 184);
    if ( v7 )
    {
      sub_76C7C0((__int64)v11);
      v16 = a2;
      v17 = a3;
      v12 = sub_8D9470;
      v15 = 0x100000001LL;
      v14 = 1;
      v18 = a4;
      sub_76CDC0(v7, (__int64)v11, v8, v9, v10);
      return v13;
    }
  }
  return result;
}

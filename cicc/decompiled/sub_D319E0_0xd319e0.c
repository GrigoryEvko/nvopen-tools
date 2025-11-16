// Function: sub_D319E0
// Address: 0xd319e0
//
__int64 __fastcall sub_D319E0(__int64 a1, __int64 a2, _WORD *a3, int a4, _QWORD **a5, _BYTE *a6, _DWORD *a7)
{
  __int64 result; // rax
  unsigned __int16 v10; // dx
  unsigned __int8 v12; // al
  _BYTE v14[96]; // [rsp+10h] [rbp-60h] BYREF

  result = 0;
  v10 = *(_WORD *)(a1 + 2);
  if ( ((v10 >> 7) & 6) == 0 && (v10 & 1) == 0 )
  {
    sub_D665A0(v14);
    v12 = sub_B46500((unsigned __int8 *)a1);
    return sub_D31270((__int64)v14, *(_QWORD *)(a1 + 8), v12, a2, a3, a4, a5, a6, a7);
  }
  return result;
}

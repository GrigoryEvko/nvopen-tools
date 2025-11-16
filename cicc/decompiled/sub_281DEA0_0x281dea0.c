// Function: sub_281DEA0
// Address: 0x281dea0
//
__int64 __fastcall sub_281DEA0(__int64 a1, _BYTE *a2, _BYTE **a3, _DWORD *a4)
{
  __int64 *v6; // rax
  _BYTE *v7; // rax
  __int64 result; // rax
  const char *v9; // [rsp+0h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-30h]
  char v11; // [rsp+21h] [rbp-2Fh]

  if ( *(_QWORD *)(a1 + 136) )
  {
    v6 = sub_CEADF0();
    v11 = 1;
    v9 = "cl::location(x) specified more than once!";
    v10 = 3;
    sub_C53280(a1, (__int64)&v9, 0, 0, (__int64)v6);
    a2 = *(_BYTE **)(a1 + 136);
  }
  else
  {
    *(_QWORD *)(a1 + 136) = a2;
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) = *a2;
  }
  v7 = *a3;
  *a2 = **a3;
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 152) = *v7;
  result = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9Fu;
  *(_BYTE *)(a1 + 12) = (32 * (*(_BYTE *)a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  return result;
}

// Function: sub_13F95E0
// Address: 0x13f95e0
//
unsigned __int64 __fastcall sub_13F95E0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 *a3,
        int a4,
        _QWORD *a5,
        _BYTE *a6,
        _DWORD *a7)
{
  unsigned __int64 result; // rax
  unsigned int v10; // edx
  unsigned __int8 v12; // al

  result = 0;
  v10 = *(unsigned __int16 *)(a1 + 18);
  if ( ((v10 >> 7) & 6) == 0 && (v10 & 1) == 0 )
  {
    v12 = sub_15F32D0(a1);
    return sub_13F9040(*(_QWORD *)(a1 - 24), *(_QWORD *)a1, v12, a2, a3, a4, a5, a6, a7);
  }
  return result;
}

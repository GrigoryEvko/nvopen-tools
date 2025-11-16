// Function: sub_23C6410
// Address: 0x23c6410
//
__int64 __fastcall sub_23C6410(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        int a7,
        int a8,
        int a9,
        char a10,
        char a11,
        char a12)
{
  char v15; // cl
  __int64 result; // rax

  *(_QWORD *)a1 = a1 + 16;
  sub_23C6360((__int64 *)a1, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  *(_QWORD *)(a1 + 32) = a1 + 48;
  sub_23C6360((__int64 *)(a1 + 32), *(_BYTE **)a3, *(_QWORD *)a3 + *(_QWORD *)(a3 + 8));
  *(_QWORD *)(a1 + 64) = a1 + 80;
  sub_23C6360((__int64 *)(a1 + 64), *(_BYTE **)a4, *(_QWORD *)a4 + *(_QWORD *)(a4 + 8));
  *(_QWORD *)(a1 + 96) = a1 + 112;
  sub_23C6360((__int64 *)(a1 + 96), *(_BYTE **)a5, *(_QWORD *)a5 + *(_QWORD *)(a5 + 8));
  v15 = a10;
  *(_DWORD *)(a1 + 128) = a7;
  *(_DWORD *)(a1 + 132) = a8;
  *(_DWORD *)(a1 + 136) = a9;
  if ( !a10 )
    v15 = (a11 ^ 1) & (a7 == 3);
  *(_BYTE *)(a1 + 142) = a12;
  *(_BYTE *)(a1 + 140) = v15;
  *(_BYTE *)(a1 + 141) = a11;
  result = *a6;
  *a6 = 0;
  *(_QWORD *)(a1 + 144) = result;
  return result;
}

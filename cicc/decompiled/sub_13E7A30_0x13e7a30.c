// Function: sub_13E7A30
// Address: 0x13e7a30
//
__int64 __fastcall sub_13E7A30(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  result = *a1;
  if ( !*a1 )
  {
    result = sub_22077B0(304);
    if ( result )
    {
      *(_QWORD *)result = 0;
      *(_QWORD *)(result + 8) = 0;
      *(_QWORD *)(result + 16) = 0;
      *(_DWORD *)(result + 24) = 0;
      *(_QWORD *)(result + 32) = 0;
      *(_QWORD *)(result + 40) = 0;
      *(_QWORD *)(result + 48) = 0;
      *(_DWORD *)(result + 56) = 0;
      *(_QWORD *)(result + 64) = 0;
      *(_QWORD *)(result + 72) = 0;
      *(_QWORD *)(result + 80) = 0;
      *(_DWORD *)(result + 88) = 0;
      *(_QWORD *)(result + 96) = result + 112;
      *(_QWORD *)(result + 104) = 0x800000000LL;
      *(_QWORD *)(result + 240) = 0;
      *(_QWORD *)(result + 248) = 0;
      *(_QWORD *)(result + 256) = 0;
      *(_DWORD *)(result + 264) = 0;
      *(_QWORD *)(result + 272) = a2;
      *(_QWORD *)(result + 280) = a3;
      *(_QWORD *)(result + 288) = a4;
      *(_QWORD *)(result + 296) = 0;
    }
    *a1 = result;
  }
  return result;
}

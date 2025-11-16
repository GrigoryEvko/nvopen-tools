// Function: sub_22C1480
// Address: 0x22c1480
//
__int64 __fastcall sub_22C1480(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rdx

  result = a1[2];
  if ( !result )
  {
    v3 = sub_B6AC80(a2, 153);
    result = sub_22077B0(0x108u);
    if ( result )
    {
      v4 = *a1;
      *(_QWORD *)result = 0;
      *(_QWORD *)(result + 64) = result + 80;
      *(_QWORD *)(result + 8) = 0;
      *(_QWORD *)(result + 16) = 0;
      *(_DWORD *)(result + 24) = 0;
      *(_QWORD *)(result + 32) = 0;
      *(_QWORD *)(result + 40) = 0;
      *(_QWORD *)(result + 48) = 0;
      *(_DWORD *)(result + 56) = 0;
      *(_QWORD *)(result + 72) = 0x800000000LL;
      *(_QWORD *)(result + 208) = 0;
      *(_QWORD *)(result + 216) = 0;
      *(_QWORD *)(result + 224) = 0;
      *(_DWORD *)(result + 232) = 0;
      *(_QWORD *)(result + 240) = v4;
      *(_QWORD *)(result + 248) = a2 + 312;
      *(_QWORD *)(result + 256) = v3;
    }
    a1[2] = result;
  }
  return result;
}

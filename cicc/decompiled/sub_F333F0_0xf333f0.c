// Function: sub_F333F0
// Address: 0xf333f0
//
__int64 __fastcall sub_F333F0(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rsi

  result = a1;
  v3 = a2[3];
  v4 = a2[4];
  v5 = *a2;
  v6 = a2[1];
  *(_BYTE *)(result + 25) = 1;
  *(_QWORD *)(result + 32) = v3;
  *(_QWORD *)result = v5;
  *(_QWORD *)(result + 8) = v6;
  *(_QWORD *)(result + 40) = v4;
  *(_BYTE *)(result + 57) = 1;
  return result;
}

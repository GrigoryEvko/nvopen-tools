// Function: sub_1F4AA00
// Address: 0x1f4aa00
//
__int64 *__fastcall sub_1F4AA00(__int64 *a1, int a2, __int64 a3, int a4, __int64 a5)
{
  __int64 v8; // rax

  v8 = sub_22077B0(32);
  if ( v8 )
  {
    *(_DWORD *)v8 = a2;
    *(_QWORD *)(v8 + 8) = a3;
    *(_DWORD *)(v8 + 16) = a4;
    *(_QWORD *)(v8 + 24) = a5;
  }
  *a1 = v8;
  a1[2] = (__int64)sub_1F4A0D0;
  a1[3] = (__int64)sub_1F4A4F0;
  return a1;
}

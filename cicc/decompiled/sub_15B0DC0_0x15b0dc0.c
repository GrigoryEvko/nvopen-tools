// Function: sub_15B0DC0
// Address: 0x15b0dc0
//
__int64 __fastcall sub_15B0DC0(
        int a1,
        int a2,
        __int64 a3,
        __int64 a4,
        char a5,
        __int64 a6,
        int a7,
        __int64 a8,
        int a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        __int64 a15,
        char a16,
        char a17,
        char a18,
        int a19)
{
  __int64 v20; // rsi
  __int64 result; // rax
  __int64 v22; // [rsp-8h] [rbp-98h]
  __int64 v23; // [rsp+0h] [rbp-90h]
  __int64 v25; // [rsp+8h] [rbp-88h]
  _QWORD v26[16]; // [rsp+10h] [rbp-80h] BYREF

  v20 = 9;
  v26[0] = a3;
  v26[3] = a8;
  v26[1] = a4;
  v26[4] = a10;
  v26[2] = a6;
  v26[5] = a11;
  v26[6] = a12;
  v26[7] = a13;
  v26[8] = a14;
  result = sub_161E980(56, 9);
  if ( result )
  {
    v23 = result;
    sub_1623D80(result, a1, 16, a19, (unsigned int)v26, 9, 0, 0);
    result = v23;
    *(_WORD *)(v23 + 2) = 17;
    *(_DWORD *)(v23 + 24) = a2;
    *(_DWORD *)(v23 + 32) = a7;
    *(_BYTE *)(v23 + 28) = a5;
    *(_DWORD *)(v23 + 36) = a9;
    *(_BYTE *)(v23 + 48) = a16;
    *(_QWORD *)(v23 + 40) = a15;
    *(_BYTE *)(v23 + 49) = a17;
    *(_BYTE *)(v23 + 50) = a18;
    v20 = v22;
  }
  if ( a19 == 1 )
  {
    v25 = result;
    sub_1621390(result, v20);
    return v25;
  }
  return result;
}

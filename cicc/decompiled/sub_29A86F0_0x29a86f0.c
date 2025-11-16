// Function: sub_29A86F0
// Address: 0x29a86f0
//
__int64 __fastcall sub_29A86F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14[5]; // [rsp+0h] [rbp-30h] BYREF

  v7 = *(_QWORD *)(a5 + 16);
  v8 = *(_QWORD *)(a5 + 32);
  v14[0] = a3;
  v14[1] = v8;
  v14[2] = v7;
  if ( (unsigned __int8)sub_D4B3D0(a3) && (unsigned __int8)sub_29A8190(v14) )
  {
    sub_22D0390(a1, a2, v10, v11, v12, v13);
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    return a1;
  }
}

// Function: sub_F703A0
// Address: 0xf703a0
//
__int64 __fastcall sub_F703A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // r14d
  __int64 v7; // r15
  char v8; // r12
  int v9; // edx
  __int16 v10; // r13
  __int64 result; // rax

  v6 = *(_DWORD *)(a1 + 104);
  v7 = *(_QWORD *)(a1 + 96);
  v8 = *(_BYTE *)(a1 + 110);
  *(_DWORD *)(a1 + 104) = *(_DWORD *)(a2 + 44);
  v9 = *(_DWORD *)(a2 + 40);
  v10 = *(_WORD *)(a1 + 108);
  if ( (unsigned int)(v9 - 17) <= 1 )
  {
    result = sub_F6FBB0(a1, a3, a2, a4);
  }
  else if ( (unsigned int)(v9 - 19) <= 1 )
  {
    result = sub_F6FD90(a1, a3, a2);
  }
  else
  {
    result = sub_F70250(a1, a3, v9, a4, a2);
  }
  *(_QWORD *)(a1 + 96) = v7;
  *(_DWORD *)(a1 + 104) = v6;
  *(_WORD *)(a1 + 108) = v10;
  *(_BYTE *)(a1 + 110) = v8;
  return result;
}

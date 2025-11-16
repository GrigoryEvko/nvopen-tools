// Function: sub_968DB0
// Address: 0x968db0
//
__int64 __fastcall sub_968DB0(_QWORD *a1, int a2, __int64 a3, unsigned __int8 a4)
{
  unsigned int v4; // r14d
  unsigned int v6; // ebx
  unsigned int v7; // r15d
  unsigned int v8; // r15d
  int v9; // eax
  char v11; // [rsp+17h] [rbp-39h] BYREF
  _QWORD v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a4;
  v6 = a2 ^ 1;
  v7 = *(_DWORD *)(a3 + 8);
  v11 = 0;
  v8 = v7 >> 8;
  if ( *a1 == sub_C33340() )
    v9 = sub_C3FF40((_DWORD)a1, (unsigned int)v12, 1, v8, v4, v6, (__int64)&v11);
  else
    v9 = sub_C34710(a1, v12, 1, v8, v4, v6, &v11);
  if ( v9 && (v9 != 16 || (_BYTE)v6) )
    return 0;
  else
    return sub_AD64C0(a3, v12[0], v4);
}

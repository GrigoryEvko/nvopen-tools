// Function: sub_2FDF4D0
// Address: 0x2fdf4d0
//
__int64 __fastcall sub_2FDF4D0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 result; // rax
  int v6; // eax
  int v7; // eax
  __int64 v8; // rcx
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 (*v12)(); // rdx

  v2 = *(_DWORD *)(a2 + 44);
  if ( (v2 & 4) == 0 && (v2 & 8) != 0 )
  {
    LOBYTE(v6) = sub_2E88A90(a2, 512, 1);
    LODWORD(v4) = v6;
    if ( !(_BYTE)v6 )
      return (unsigned int)v4;
  }
  else
  {
    v3 = *(_QWORD *)(a2 + 16);
    v4 = (*(_QWORD *)(v3 + 24) >> 9) & 1LL;
    if ( (*(_QWORD *)(v3 + 24) & 0x200LL) == 0 )
      return (unsigned int)v4;
  }
  v7 = *(_DWORD *)(a2 + 44);
  v8 = v7 & 0xFFFFFF;
  v9 = v7 & 4;
  if ( (v7 & 4) == 0 && (v7 & 8) != 0 )
  {
    LOBYTE(v10) = sub_2E88A90(a2, 1024, 1);
    v9 = *(_DWORD *)(a2 + 44) & 4;
    v8 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  }
  else
  {
    v10 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 10) & 1LL;
  }
  if ( (_BYTE)v10 )
  {
    if ( v9 || (v8 & 8) == 0 )
      v11 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 8) & 1LL;
    else
      LOBYTE(v11) = sub_2E88A90(a2, 256, 1);
    if ( !(_BYTE)v11 )
      return (unsigned int)v4;
    v9 = *(_DWORD *)(a2 + 44) & 4;
    v8 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  }
  if ( !v9 && (v8 &= 8u, (_DWORD)v8) )
    LOBYTE(result) = sub_2E88A90(a2, (__int64)&dword_400000, 2);
  else
    result = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 22) & 1LL;
  if ( !(_BYTE)result )
    return (unsigned int)v4;
  v12 = *(__int64 (**)())(*(_QWORD *)a1 + 920LL);
  if ( v12 != sub_2DB1B30 )
    LODWORD(result) = ((__int64 (__fastcall *)(__int64, __int64, __int64 (*)(), __int64))v12)(a1, a2, v12, v8) ^ 1;
  return (unsigned int)result;
}

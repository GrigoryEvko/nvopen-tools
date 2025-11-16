// Function: sub_1F7DDA0
// Address: 0x1f7dda0
//
__int64 __fastcall sub_1F7DDA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned __int64 v4; // rdi
  int v5; // edx
  __int64 v6; // rdx
  char v7; // r8
  __int64 v8; // rdx
  bool v9; // zf
  int v10; // edx

  result = a1;
  v4 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 )
  {
    v6 = *(_QWORD *)(a2 + 8) + a3;
    v7 = *(_BYTE *)(a2 + 16);
    if ( (*(_QWORD *)a2 & 4) != 0 )
    {
      *(_QWORD *)(result + 8) = v6;
      v10 = *(_DWORD *)(v4 + 12);
      *(_BYTE *)(result + 16) = v7;
      *(_QWORD *)result = v4 | 4;
      *(_DWORD *)(result + 20) = v10;
    }
    else
    {
      *(_QWORD *)(result + 8) = v6;
      v8 = *(_QWORD *)v4;
      *(_QWORD *)result = v4;
      v9 = *(_BYTE *)(v8 + 8) == 16;
      *(_BYTE *)(result + 16) = v7;
      if ( v9 )
        v8 = **(_QWORD **)(v8 + 16);
      *(_DWORD *)(result + 20) = *(_DWORD *)(v8 + 8) >> 8;
    }
  }
  else
  {
    v5 = *(_DWORD *)(a2 + 20);
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)result = 0;
    *(_QWORD *)(result + 8) = 0;
    *(_DWORD *)(result + 20) = v5;
  }
  return result;
}

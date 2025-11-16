// Function: sub_B4C1E0
// Address: 0xb4c1e0
//
__int64 __fastcall sub_B4C1E0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx

  *(_DWORD *)(a1 + 72) = a4;
  *(_DWORD *)(a1 + 4) = (1 - ((a3 == 0) - 1)) | *(_DWORD *)(a1 + 4) & 0xF8000000;
  sub_BD2A10(a1, a4, 0);
  result = *(_QWORD *)(a1 - 8);
  if ( *(_QWORD *)result )
  {
    v6 = *(_QWORD *)(result + 8);
    **(_QWORD **)(result + 16) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(result + 16);
  }
  *(_QWORD *)result = a2;
  if ( a2 )
  {
    v7 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(result + 8) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = result + 8;
    *(_QWORD *)(result + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = result;
  }
  if ( a3 )
  {
    result = *(_QWORD *)(a1 - 8);
    *(_WORD *)(a1 + 2) |= 1u;
    if ( *(_QWORD *)(result + 32) )
    {
      v8 = *(_QWORD *)(result + 40);
      **(_QWORD **)(result + 48) = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 16) = *(_QWORD *)(result + 48);
    }
    *(_QWORD *)(result + 32) = a3;
    v9 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(result + 40) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = result + 40;
    *(_QWORD *)(result + 48) = a3 + 16;
    *(_QWORD *)(a3 + 16) = result + 32;
  }
  return result;
}

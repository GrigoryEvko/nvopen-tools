// Function: sub_AC43C0
// Address: 0xac43c0
//
__int64 __fastcall sub_AC43C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool v8; // zf
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rax

  sub_BD35F0(a1, *(_QWORD *)(a2 + 8), 8);
  v8 = *(_QWORD *)(a1 - 128) == 0;
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0x38000000 | 4;
  if ( !v8 )
  {
    v9 = *(_QWORD *)(a1 - 120);
    **(_QWORD **)(a1 - 112) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(a1 - 112);
  }
  result = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 - 128) = a2;
  *(_QWORD *)(a1 - 120) = result;
  if ( result )
    *(_QWORD *)(result + 16) = a1 - 120;
  v8 = *(_QWORD *)(a1 - 96) == 0;
  *(_QWORD *)(a1 - 112) = a2 + 16;
  *(_QWORD *)(a2 + 16) = a1 - 128;
  if ( !v8 )
  {
    result = *(_QWORD *)(a1 - 88);
    **(_QWORD **)(a1 - 80) = result;
    if ( result )
      *(_QWORD *)(result + 16) = *(_QWORD *)(a1 - 80);
  }
  *(_QWORD *)(a1 - 96) = a3;
  if ( a3 )
  {
    v11 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 88) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = a1 - 88;
    result = a1 - 96;
    *(_QWORD *)(a1 - 80) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 96;
  }
  if ( *(_QWORD *)(a1 - 64) )
  {
    result = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = result;
    if ( result )
      *(_QWORD *)(result + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a4;
  if ( a4 )
  {
    v12 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(a1 - 56) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = a1 - 56;
    result = a1 - 64;
    *(_QWORD *)(a1 - 48) = a4 + 16;
    *(_QWORD *)(a4 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    result = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = result;
    if ( result )
      *(_QWORD *)(result + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a5;
  if ( a5 )
  {
    result = *(_QWORD *)(a5 + 16);
    *(_QWORD *)(a1 - 24) = result;
    if ( result )
      *(_QWORD *)(result + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a5 + 16;
    *(_QWORD *)(a5 + 16) = a1 - 32;
  }
  return result;
}

// Function: sub_B53980
// Address: 0xb53980
//
__int64 __fastcall sub_B53980(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx

  v5 = *(_DWORD *)(a1 + 4);
  *(_DWORD *)(a1 + 72) = a4;
  *(_DWORD *)(a1 + 4) = v5 & 0xF8000000 | 2;
  sub_BD2A10(a1, a4, 0);
  v6 = *(_QWORD *)(a1 - 8);
  if ( *(_QWORD *)v6 )
  {
    v7 = *(_QWORD *)(v6 + 8);
    **(_QWORD **)(v6 + 16) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = *(_QWORD *)(v6 + 16);
  }
  *(_QWORD *)v6 = a2;
  if ( a2 )
  {
    v8 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(v6 + 8) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = v6 + 8;
    *(_QWORD *)(v6 + 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v6;
  }
  result = *(_QWORD *)(a1 - 8);
  if ( *(_QWORD *)(result + 32) )
  {
    v10 = *(_QWORD *)(result + 40);
    **(_QWORD **)(result + 48) = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(result + 48);
  }
  *(_QWORD *)(result + 32) = a3;
  if ( a3 )
  {
    v11 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(result + 40) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = result + 40;
    *(_QWORD *)(result + 48) = a3 + 16;
    result += 32;
    *(_QWORD *)(a3 + 16) = result;
  }
  return result;
}

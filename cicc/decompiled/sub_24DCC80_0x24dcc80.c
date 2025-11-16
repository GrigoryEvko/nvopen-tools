// Function: sub_24DCC80
// Address: 0x24dcc80
//
__int64 __fastcall sub_24DCC80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx
  int v6; // eax
  __int64 result; // rax

  v3 = sub_24F3110(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a3, a2);
  if ( *(_QWORD *)(a2 - 32) )
  {
    v4 = *(_QWORD *)(a2 - 24);
    **(_QWORD **)(a2 - 16) = v4;
    if ( v4 )
      *(_QWORD *)(v4 + 16) = *(_QWORD *)(a2 - 16);
  }
  *(_QWORD *)(a2 - 32) = v3;
  if ( v3 )
  {
    v5 = *(_QWORD *)(v3 + 16);
    *(_QWORD *)(a2 - 24) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = a2 - 24;
    *(_QWORD *)(a2 - 16) = v3 + 16;
    *(_QWORD *)(v3 + 16) = a2 - 32;
  }
  v6 = *(unsigned __int16 *)(a2 + 2);
  LOWORD(v6) = v6 & 0xF003;
  result = v6 | 0x20u;
  *(_WORD *)(a2 + 2) = result;
  return result;
}

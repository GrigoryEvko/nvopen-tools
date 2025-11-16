// Function: sub_1195EB0
// Address: 0x1195eb0
//
__int64 __fastcall sub_1195EB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  _BYTE *v10; // rax

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) || *(_BYTE *)a2 != 57 )
    return 0;
  v4 = *(_QWORD *)(a2 - 64);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 )
    return 0;
  if ( *(_QWORD *)(v5 + 8) )
    return 0;
  if ( (unsigned __int8)(*(_BYTE *)v4 - 55) > 1u )
    return 0;
  v7 = *(_QWORD *)sub_986520(v4);
  if ( !v7 )
    return 0;
  **(_QWORD **)a1 = v7;
  if ( *(_QWORD *)(sub_986520(v4) + 32) != *(_QWORD *)(a1 + 8) )
    return 0;
  v8 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v8 == 17 )
  {
    **(_QWORD **)(a1 + 16) = v8 + 24;
    return 1;
  }
  v9 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
  if ( (unsigned int)v9 > 1 )
    return 0;
  if ( *(_BYTE *)v8 > 0x15u )
    return 0;
  v10 = sub_AD7630(v8, *(unsigned __int8 *)(a1 + 24), v9);
  if ( !v10 || *v10 != 17 )
    return 0;
  **(_QWORD **)(a1 + 16) = v10 + 24;
  return 1;
}

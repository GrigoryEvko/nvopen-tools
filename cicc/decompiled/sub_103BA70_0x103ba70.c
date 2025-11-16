// Function: sub_103BA70
// Address: 0x103ba70
//
__int64 __fastcall sub_103BA70(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 result; // rax
  int v7; // ecx
  __int64 v8; // rax

  v2 = *(_QWORD *)(a1 - 32);
  v3 = a1 - 32;
  if ( *(_BYTE *)a1 == 27 )
  {
    if ( v2 )
    {
      v4 = *(_QWORD *)(a1 - 24);
      **(_QWORD **)(a1 - 16) = v4;
      if ( v4 )
        *(_QWORD *)(v4 + 16) = *(_QWORD *)(a1 - 16);
    }
    *(_QWORD *)(a1 - 32) = a2;
    if ( a2 )
    {
      v5 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a1 - 24) = v5;
      if ( v5 )
        *(_QWORD *)(v5 + 16) = a1 - 24;
      *(_QWORD *)(a1 - 16) = a2 + 16;
      *(_QWORD *)(a2 + 16) = v3;
    }
    if ( *(_BYTE *)a2 == 27 )
      result = *(unsigned int *)(a2 + 80);
    else
      result = *(unsigned int *)(a2 + 72);
    *(_DWORD *)(a1 + 84) = result;
  }
  else
  {
    if ( *(_BYTE *)a2 == 27 )
      v7 = *(_DWORD *)(a2 + 80);
    else
      v7 = *(_DWORD *)(a2 + 72);
    *(_DWORD *)(a1 + 80) = v7;
    if ( v2 )
    {
      v8 = *(_QWORD *)(a1 - 24);
      **(_QWORD **)(a1 - 16) = v8;
      if ( v8 )
        *(_QWORD *)(v8 + 16) = *(_QWORD *)(a1 - 16);
    }
    *(_QWORD *)(a1 - 32) = a2;
    result = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 - 24) = result;
    if ( result )
      *(_QWORD *)(result + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a2 + 16;
    *(_QWORD *)(a2 + 16) = v3;
  }
  return result;
}

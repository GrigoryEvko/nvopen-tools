// Function: sub_B19720
// Address: 0xb19720
//
__int64 __fastcall sub_B19720(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // ecx
  __int64 v4; // rcx
  unsigned int v5; // eax
  unsigned int v6; // edx
  __int64 v7; // r8
  __int64 v8; // rcx
  unsigned int v9; // eax
  __int64 v10; // rsi

  if ( a2 == a3 )
    return 1;
  if ( a3 )
  {
    v4 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
    v5 = *(_DWORD *)(a3 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = *(_DWORD *)(a1 + 32);
  v7 = 0;
  if ( v5 < v6 )
    v7 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v4);
  if ( a2 )
  {
    v8 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v9 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v8 = 0;
    v9 = 0;
  }
  v10 = 0;
  if ( v9 < v6 )
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v8);
  if ( v7 == v10 || v7 == 0 )
    goto LABEL_30;
  if ( !v10 )
    goto LABEL_31;
  if ( v10 == *(_QWORD *)(v7 + 8) )
LABEL_30:
    JUMPOUT(0xB19640);
  if ( v7 == *(_QWORD *)(v10 + 8) || *(_DWORD *)(v10 + 16) >= *(_DWORD *)(v7 + 16) )
LABEL_31:
    JUMPOUT(0xB19645);
  if ( *(_BYTE *)(a1 + 112) )
    JUMPOUT(0xB19630);
  v3 = *(_DWORD *)(a1 + 116) + 1;
  *(_DWORD *)(a1 + 116) = v3;
  if ( v3 <= 0x20 )
    JUMPOUT(0xB1961D);
  return sub_B19658(a1, v10, v7);
}

// Function: sub_15690E0
// Address: 0x15690e0
//
__int64 __fastcall sub_15690E0(int a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rdx
  char v5; // al
  char v6; // al
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax

  if ( a1 == 47 )
  {
    v4 = *a2;
    v5 = *(_BYTE *)(*a2 + 8);
    if ( v5 == 16 )
    {
      v4 = **(_QWORD **)(v4 + 16);
      if ( *(_BYTE *)(v4 + 8) == 15 )
      {
LABEL_4:
        v6 = *(_BYTE *)(a3 + 8);
        v7 = a3;
        if ( v6 == 16 )
        {
          v7 = **(_QWORD **)(a3 + 16);
          v6 = *(_BYTE *)(v7 + 8);
        }
        if ( v6 == 15 && *(_DWORD *)(v7 + 8) >> 8 != *(_DWORD *)(v4 + 8) >> 8 )
        {
          v8 = sub_16498A0(a2);
          v9 = sub_1643360(v8);
          v10 = sub_15A4180(a2, v9, 0);
          return sub_15A3BA0(v10, a3, 0);
        }
      }
    }
    else if ( v5 == 15 )
    {
      goto LABEL_4;
    }
    return 0;
  }
  return 0;
}

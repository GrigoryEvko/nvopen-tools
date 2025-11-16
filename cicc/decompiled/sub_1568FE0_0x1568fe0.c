// Function: sub_1568FE0
// Address: 0x1568fe0
//
__int64 __fastcall sub_1568FE0(int a1, __int64 *a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // rdx
  char v7; // al
  char v8; // al
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _BYTE v14[16]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v15; // [rsp-38h] [rbp-38h]

  if ( a1 == 47 )
  {
    *a4 = 0;
    v6 = *a2;
    v7 = *(_BYTE *)(*a2 + 8);
    if ( v7 == 16 )
    {
      v6 = **(_QWORD **)(v6 + 16);
      if ( *(_BYTE *)(v6 + 8) == 15 )
      {
LABEL_4:
        v8 = *(_BYTE *)(a3 + 8);
        v9 = a3;
        if ( v8 == 16 )
        {
          v9 = **(_QWORD **)(a3 + 16);
          v8 = *(_BYTE *)(v9 + 8);
        }
        if ( v8 == 15 && *(_DWORD *)(v9 + 8) >> 8 != *(_DWORD *)(v6 + 8) >> 8 )
        {
          v10 = sub_16498A0(a2);
          v11 = sub_1643360(v10);
          v15 = 257;
          v12 = sub_15FDBD0(45, a2, v11, v14, 0);
          *a4 = v12;
          v15 = 257;
          return sub_15FDBD0(46, v12, a3, v14, 0);
        }
      }
    }
    else if ( v7 == 15 )
    {
      goto LABEL_4;
    }
    return 0;
  }
  return 0;
}

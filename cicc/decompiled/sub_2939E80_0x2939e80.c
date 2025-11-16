// Function: sub_2939E80
// Address: 0x2939e80
//
__int64 __fastcall sub_2939E80(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  __int64 v4; // r14
  unsigned int v6; // ecx
  unsigned int v7; // eax
  unsigned int v8; // eax
  unsigned int v9; // r13d
  __int64 v10; // rcx
  int v11; // edx
  __int64 v12; // rdx
  __int64 v14; // [rsp+0h] [rbp-40h]
  unsigned int v15; // [rsp+Ch] [rbp-34h]

  if ( *(_BYTE *)(a3 + 8) != 17 )
    goto LABEL_13;
  v3 = *(_DWORD *)(a3 + 32);
  v4 = *(_QWORD *)(a3 + 24);
  if ( v3 == 1 )
  {
    v9 = 1;
    v12 = 0;
  }
  else
  {
    if ( *(_BYTE *)(v4 + 8) != 14 )
    {
      v6 = sub_BCB060(*(_QWORD *)(a3 + 24));
      v7 = *(_DWORD *)(a2 + 1132);
      if ( 2 * v6 <= v7 )
      {
        v8 = v7 / v6;
        v9 = v8;
        if ( v8 < v3 )
        {
          v15 = (v3 != 0) + (v3 - (v3 != 0)) / v8;
          v10 = sub_BCDA70((__int64 *)v4, v8);
          v11 = v3 % v9;
          if ( v3 % v9 > 1 )
          {
            v14 = v10;
            v3 = v15;
            v12 = sub_BCDA70((__int64 *)v4, v11);
            v4 = v14;
          }
          else if ( v3 % v9 == 1 )
          {
            v12 = v4;
            v3 = v15;
            v4 = v10;
          }
          else
          {
            v3 = v15;
            v4 = v10;
            v12 = 0;
          }
          goto LABEL_10;
        }
LABEL_13:
        *(_QWORD *)(a1 + 32) = 0;
        return a1;
      }
    }
    v12 = 0;
    v9 = 1;
  }
LABEL_10:
  *(_QWORD *)a1 = a3;
  *(_DWORD *)(a1 + 8) = v9;
  *(_DWORD *)(a1 + 12) = v3;
  *(_QWORD *)(a1 + 16) = v4;
  *(_QWORD *)(a1 + 24) = v12;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}

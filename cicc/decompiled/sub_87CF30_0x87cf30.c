// Function: sub_87CF30
// Address: 0x87cf30
//
__int64 __fastcall sub_87CF30(
        __int64 a1,
        unsigned int a2,
        int a3,
        FILE *a4,
        __int64 a5,
        _DWORD *a6,
        int a7,
        unsigned int a8,
        int a9,
        unsigned int a10,
        _DWORD *a11)
{
  __int64 v13; // r13
  __int64 *v15; // r9
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 result; // rax
  __int64 v19; // rax
  int v20; // eax
  FILE *v21; // rsi
  __int64 v22; // [rsp+8h] [rbp-48h]
  int v23; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v24; // [rsp+18h] [rbp-38h] BYREF

  v13 = a5;
  v24 = 0;
  if ( a11 )
    *a11 = 0;
  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  if ( a5 )
  {
    while ( *(_BYTE *)(v13 + 140) == 12 )
      v13 = *(_QWORD *)(v13 + 160);
  }
  v15 = &v24;
  if ( a11 )
    v15 = 0;
  v16 = sub_697AE0(a1, a2, a3, (int)a4, (int)&v23, (__int64)v15, (__int64)a6);
  v17 = v16;
  if ( *a6 )
  {
    sub_876E10(a1, v13, a4, a10, 0, a11);
    return 0;
  }
  if ( v23 )
  {
    if ( !a11 )
    {
      sub_685360(0x122u, a4, a1);
      return 0;
    }
    goto LABEL_14;
  }
  if ( !v16 )
    goto LABEL_38;
  if ( (*(_BYTE *)(v16 + 104) & 1) != 0 )
  {
    v22 = v16;
    v20 = sub_8796F0(v16);
    v17 = v22;
  }
  else
  {
    v19 = *(_QWORD *)(v16 + 88);
    if ( *(_BYTE *)(v17 + 80) == 20 )
      v19 = *(_QWORD *)(v19 + 176);
    v20 = (*(_BYTE *)(v19 + 208) & 4) != 0;
  }
  if ( v20 )
  {
LABEL_38:
    if ( (*(_BYTE *)(a1 + 180) & 4) == 0 || (result = 0, !a9) )
    {
      if ( a2 != 1 || v24 )
      {
        if ( !a11 )
        {
          v21 = (FILE *)sub_67DA80(0x14Eu, a4, a1);
          sub_87CA90(v24, v21);
          sub_685910((__int64)v21, v21);
          return 0;
        }
      }
      else if ( !a11 )
      {
        sub_685360(0x14Cu, a4, a1);
        return 0;
      }
LABEL_14:
      *a11 = 1;
      return 0;
    }
  }
  else
  {
    if ( a7 )
      sub_8769C0(v17, a4, v13, 0, a8, 1, a10, 0, a11);
    return *(_QWORD *)(v17 + 88);
  }
  return result;
}

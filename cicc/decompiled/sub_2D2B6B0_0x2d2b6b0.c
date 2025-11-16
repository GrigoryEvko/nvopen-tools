// Function: sub_2D2B6B0
// Address: 0x2d2b6b0
//
__int64 __fastcall sub_2D2B6B0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v3; // r9
  int v4; // r13d
  __int64 v5; // rcx
  __int64 v6; // r12
  unsigned int i; // ebx
  __int64 v8; // rdx
  _QWORD *v9; // r15
  int v11; // eax
  bool v12; // al
  unsigned int v13; // ebx
  int v14; // [rsp+Ch] [rbp-74h]
  _QWORD *v15; // [rsp+10h] [rbp-70h]
  __int64 v17; // [rsp+20h] [rbp-60h]
  __int64 v18; // [rsp+28h] [rbp-58h]
  _QWORD v19[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v20[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v3 = a1 + 16;
    v4 = 3;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 24);
    v3 = *(_QWORD *)(a1 + 16);
    v4 = v11 - 1;
    if ( !v11 )
    {
      *a3 = 0;
      return 0;
    }
  }
  v5 = *a2;
  v6 = a2[1];
  v19[0] = -1;
  v19[1] = -1;
  v20[0] = -2;
  v20[1] = -2;
  v14 = 1;
  v15 = 0;
  for ( i = v4 & ((unsigned __int16)v6 | ((_DWORD)v5 << 16)); ; i = v4 & v13 )
  {
    v8 = 16LL * i;
    v9 = (_QWORD *)(v3 + v8);
    if ( v5 == *(_QWORD *)(v3 + v8) && v6 == v9[1] )
    {
      *a3 = v9;
      return 1;
    }
    v17 = v5;
    v18 = v3;
    if ( sub_2D27C10((_QWORD *)(v3 + v8), v19) )
      break;
    v12 = sub_2D27C10(v9, v20);
    v3 = v18;
    v5 = v17;
    if ( !v15 )
    {
      if ( !v12 )
        v9 = 0;
      v15 = v9;
    }
    v13 = v14 + i;
    ++v14;
  }
  if ( v15 )
    v9 = v15;
  *a3 = v9;
  return 0;
}

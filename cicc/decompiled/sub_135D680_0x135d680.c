// Function: sub_135D680
// Address: 0x135d680
//
__int64 __fastcall sub_135D680(_QWORD *a1, __int64 a2, unsigned int a3)
{
  _QWORD *v3; // r15
  int v4; // r12d
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 result; // rax
  bool v9; // cc
  _QWORD *v10; // rax
  char v11; // [rsp+Fh] [rbp-41h]
  unsigned int v12[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = a1;
  v4 = 0;
  v11 = a3 & 1;
  v6 = *a1;
  if ( *(_BYTE *)(*a1 + 8LL) != 15 )
    return 15;
  while ( 1 )
  {
    v7 = *(_DWORD *)(v6 + 8) >> 8;
    if ( v7 == 4 )
      break;
    if ( v7 > 4 )
    {
      if ( v7 == 5 )
        return 8;
      if ( v7 == 101 )
        return 16;
    }
    else
    {
      if ( v7 == 1 )
        return 1;
      if ( v7 == 3 )
        return 2;
    }
    v9 = *((_BYTE *)v3 + 16) <= 0x17u;
    v12[0] = 15;
    if ( !v9 )
    {
      sub_1CCB380(v3, v12);
      result = v12[0];
      if ( v12[0] != 15 )
        return result;
    }
    v10 = (_QWORD *)sub_14AD280(v3, a2, 1);
    if ( v10 != v3 && (a3 > ++v4 || !v11) )
    {
      v3 = v10;
      v6 = *v10;
      if ( *(_BYTE *)(v6 + 8) == 15 )
        continue;
    }
    return 15;
  }
  return 4;
}

// Function: sub_2753D10
// Address: 0x2753d10
//
bool __fastcall sub_2753D10(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 *v2; // rax
  __int64 v3; // rsi
  unsigned __int8 *v4; // r12
  bool result; // al
  __int64 v6; // rcx
  unsigned __int8 *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  int v11; // eax
  int v12; // ecx
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r8
  int v16; // eax
  int v17; // r9d

  v2 = sub_BD3990(a2, (__int64)a2);
  v3 = *v2;
  v4 = v2;
  if ( (unsigned __int8)v3 <= 0x1Cu )
  {
    if ( (_BYTE)v3 != 5 || *((_WORD *)v2 + 1) != 34 )
      return 1;
  }
  else if ( (_BYTE)v3 != 63 )
  {
    result = sub_AA5B70(*((_QWORD *)v2 + 5));
    if ( result )
      return 1;
    goto LABEL_14;
  }
  v6 = *((_DWORD *)v2 + 1) & 0x7FFFFFF;
  v7 = &v2[32 * (1 - v6)];
  if ( v4 == v7 )
  {
LABEL_19:
    v4 = sub_BD3990(*(unsigned __int8 **)&v4[-32 * v6], v3);
    if ( *v4 <= 0x1Cu )
      return 1;
  }
  else
  {
    while ( **(_BYTE **)v7 == 17 )
    {
      v7 += 32;
      if ( v4 == v7 )
        goto LABEL_19;
    }
    if ( (unsigned __int8)v3 <= 0x1Cu )
      return 1;
  }
  result = sub_AA5B70(*((_QWORD *)v4 + 5));
  if ( result )
    return 1;
LABEL_14:
  if ( !*(_BYTE *)(a1 + 832) )
  {
    v8 = *(_QWORD *)(a1 + 824);
    v9 = *((_QWORD *)v4 + 5);
    v10 = *(_QWORD *)(v8 + 8);
    v11 = *(_DWORD *)(v8 + 24);
    if ( v11 )
    {
      v12 = v11 - 1;
      v13 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v14 = (__int64 *)(v10 + 16LL * v13);
      v15 = *v14;
      if ( v9 == *v14 )
      {
LABEL_17:
        if ( v14[1] )
          return 0;
      }
      else
      {
        v16 = 1;
        while ( v15 != -4096 )
        {
          v17 = v16 + 1;
          v13 = v12 & (v16 + v13);
          v14 = (__int64 *)(v10 + 16LL * v13);
          v15 = *v14;
          if ( v9 == *v14 )
            goto LABEL_17;
          v16 = v17;
        }
      }
    }
    return 1;
  }
  return result;
}

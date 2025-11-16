// Function: sub_2B608C0
// Address: 0x2b608c0
//
char __fastcall sub_2B608C0(__int64 a1, char *a2)
{
  char v2; // dl
  char result; // al
  __int64 v4; // rax
  char v5; // cl
  __int64 v6; // r8
  __int64 *v7; // r9
  int v8; // r10d
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  char v12; // cl
  char v13; // cl
  __int64 v14[2]; // [rsp+0h] [rbp-10h] BYREF

  v2 = *a2;
  if ( *(_BYTE *)a1 == 63 )
  {
    if ( v2 == 63 )
    {
      if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 2 || (*((_DWORD *)a2 + 1) & 0x7FFFFFF) != 2 )
        return 0;
      if ( (unsigned __int8)sub_2B0D8B0(*(unsigned __int8 **)(a1 - 32)) )
      {
        v4 = 2;
        goto LABEL_6;
      }
      if ( v13 )
      {
LABEL_10:
        v9 = *(_QWORD *)(v6 + 32 * (1LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
        v14[1] = *(_QWORD *)&a2[32 * (1LL - (v8 & 0x7FFFFFF))];
        v14[0] = v9;
        if ( sub_2B5F980(v14, 2u, v7) )
          return v10 != 0;
        return 0;
      }
    }
    else
    {
      v11 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
      if ( (_DWORD)v11 != 2 )
        return 0;
      result = sub_2B0D8B0(*(unsigned __int8 **)(a1 + 32 * (1 - v11)));
      if ( result )
        return result;
      if ( v12 )
        return 0;
    }
    return 1;
  }
  result = 1;
  if ( v2 != 63 )
    return result;
  v4 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  if ( (_DWORD)v4 != 2 )
    return 0;
LABEL_6:
  result = sub_2B0D8B0(*(unsigned __int8 **)&a2[32 * (1 - v4)]);
  if ( !result )
  {
    result = 1;
    if ( v5 )
    {
      if ( !v6 || !a2 )
        return 0;
      goto LABEL_10;
    }
  }
  return result;
}

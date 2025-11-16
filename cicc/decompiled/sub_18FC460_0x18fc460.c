// Function: sub_18FC460
// Address: 0x18fc460
//
char __fastcall sub_18FC460(char *a1, __int64 a2)
{
  char v2; // dl
  __int64 v4; // rdi
  __int64 v5; // rcx
  unsigned __int8 v6; // al
  char result; // al
  __int64 v8; // r10
  __int64 v9; // rdi
  unsigned __int8 v10; // r8
  int v11; // edx
  __int64 v12; // r8
  int v13; // r8d
  __int64 v14; // rax
  int v15; // eax

  v2 = *a1;
  if ( *a1 )
  {
    v5 = *((_QWORD *)a1 + 1);
    goto LABEL_6;
  }
  v4 = *((_QWORD *)a1 + 4);
  v5 = 0;
  v6 = *(_BYTE *)(v4 + 16);
  if ( v6 <= 0x17u )
    goto LABEL_6;
  if ( v6 != 54 && v6 != 55 )
  {
    if ( v6 == 78 )
    {
      v14 = *(_QWORD *)(v4 - 24);
      if ( !*(_BYTE *)(v14 + 16) )
      {
        v15 = *(_DWORD *)(v14 + 36);
        if ( v15 == 4085 || v15 == 4057 )
        {
          v5 = *(_QWORD *)(v4 + 24 * (1LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
        }
        else if ( v15 == 4503 || v15 == 4492 )
        {
          v5 = *(_QWORD *)(v4 + 24 * (2LL - (*(_DWORD *)(v4 + 20) & 0xFFFFFFF)));
        }
      }
    }
LABEL_6:
    result = *(_BYTE *)a2;
    if ( !*(_BYTE *)a2 )
      goto LABEL_7;
LABEL_18:
    v9 = *(_QWORD *)(a2 + 8);
    goto LABEL_11;
  }
  result = *(_BYTE *)a2;
  v5 = *(_QWORD *)(v4 - 24);
  if ( *(_BYTE *)a2 )
    goto LABEL_18;
LABEL_7:
  v8 = *(_QWORD *)(a2 + 32);
  v9 = 0;
  v10 = *(_BYTE *)(v8 + 16);
  if ( v10 > 0x17u )
  {
    if ( v10 == 54 || v10 == 55 )
    {
      if ( *(_QWORD *)(v8 - 24) != v5 )
        return 0;
      goto LABEL_14;
    }
    if ( v10 == 78 )
    {
      v12 = *(_QWORD *)(v8 - 24);
      if ( !*(_BYTE *)(v12 + 16) )
      {
        v13 = *(_DWORD *)(v12 + 36);
        if ( v13 == 4085 || v13 == 4057 )
        {
          v9 = *(_QWORD *)(v8 + 24 * (1LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)));
        }
        else if ( v13 == 4503 || v13 == 4492 )
        {
          v9 = *(_QWORD *)(v8 + 24 * (2LL - (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)));
        }
      }
    }
  }
LABEL_11:
  if ( v9 != v5 )
    return 0;
LABEL_14:
  if ( v2 )
  {
    v11 = *((unsigned __int16 *)a1 + 10);
    if ( result )
      return *(unsigned __int16 *)(a2 + 20) == v11;
  }
  else
  {
    v11 = -1;
    if ( result )
      return *(unsigned __int16 *)(a2 + 20) == v11;
    return 1;
  }
  return result;
}

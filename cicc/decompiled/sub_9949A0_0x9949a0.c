// Function: sub_9949A0
// Address: 0x9949a0
//
__int64 __fastcall sub_9949A0(__int64 a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v4; // rax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // r13
  __int16 v12; // ax
  int v13; // eax
  _BYTE *v14; // rax

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 == 85 )
  {
    v6 = *((_QWORD *)a2 - 4);
    if ( v6
      && !*(_BYTE *)v6
      && *(_QWORD *)(v6 + 24) == *((_QWORD *)a2 + 10)
      && (*(_BYTE *)(v6 + 33) & 0x20) != 0
      && *(_DWORD *)(v6 + 36) == 365
      && *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)] == *(_QWORD *)a1 )
    {
      v7 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
      if ( *(_BYTE *)v7 == 17 )
      {
        **(_QWORD **)(a1 + 8) = v7 + 24;
        return 1;
      }
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v7 <= 0x15u )
      {
        v14 = (_BYTE *)sub_AD7630(v7, *(unsigned __int8 *)(a1 + 16));
        if ( v14 )
        {
          if ( *v14 == 17 )
          {
            **(_QWORD **)(a1 + 8) = v14 + 24;
            return 1;
          }
        }
      }
    }
  }
  else
  {
    if ( v2 != 86 )
      return 0;
    v4 = *((_QWORD *)a2 - 12);
    if ( *(_BYTE *)v4 != 82 )
      return 0;
    v8 = *((_QWORD *)a2 - 8);
    v9 = *(_QWORD *)(v4 - 64);
    v10 = *((_QWORD *)a2 - 4);
    v11 = *(_QWORD *)(v4 - 32);
    if ( v9 == v8 && v11 == v10 )
    {
      v12 = *(_WORD *)(v4 + 2);
      goto LABEL_19;
    }
    if ( v11 == v8 && v9 == v10 )
    {
      v12 = *(_WORD *)(v4 + 2);
      if ( v9 != v8 )
      {
        v13 = sub_B52870(v12 & 0x3F);
        goto LABEL_20;
      }
LABEL_19:
      v13 = v12 & 0x3F;
LABEL_20:
      if ( (unsigned int)(v13 - 34) <= 1 && v9 == *(_QWORD *)a1 )
        return sub_991580(a1 + 8, v11);
    }
  }
  return 0;
}

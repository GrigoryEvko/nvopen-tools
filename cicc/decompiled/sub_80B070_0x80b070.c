// Function: sub_80B070
// Address: 0x80b070
//
const char *__fastcall sub_80B070(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rax
  char *v5; // r13
  __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v9; // rax
  __int64 v10; // rax

  v3 = *(_QWORD *)(a1 + 40);
  if ( v3 )
  {
    if ( *(_BYTE *)(v3 + 28) == 3 )
    {
      v9 = *(_QWORD *)(v3 + 32);
      if ( v9 )
      {
        if ( (*(_BYTE *)(v9 + 89) & 0x40) != 0
          || ((*(_BYTE *)(v9 + 89) & 8) != 0 ? (v10 = *(_QWORD *)(v9 + 24)) : (v10 = *(_QWORD *)(v9 + 8)), !v10) )
        {
          v6 = 12;
          v5 = (char *)byte_3F871B3;
          goto LABEL_10;
        }
      }
    }
  }
  if ( !dword_4F18BD8 )
  {
    if ( *(_QWORD *)a1 )
      v4 = sub_72B7A0((_QWORD *)a1);
    else
      v4 = (_QWORD *)qword_4D03FF0;
    v5 = *(char **)v4[49];
    if ( !v5 )
      v5 = sub_723F40(0);
    if ( *(_DWORD *)(a2 + 48) )
      return byte_3F871B3;
    v6 = strlen(v5) + 12;
LABEL_10:
    v7 = sub_7E1510(v6);
    *(_QWORD *)v7 = 0x5F4C41424F4C475FLL;
    *(_DWORD *)(v7 + 8) = (_DWORD)&loc_5F4E5F;
    strcpy((char *)(v7 + 11), v5);
    *(_BYTE *)(a1 + 89) |= 0x48u;
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 24) = v7;
    return (const char *)v7;
  }
  *(_DWORD *)(a2 + 48) = 1;
  return byte_3F871B3;
}

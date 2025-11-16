// Function: sub_2650F70
// Address: 0x2650f70
//
bool __fastcall sub_2650F70(__int64 *a1, __int64 a2, __int64 a3)
{
  char *v5; // rcx
  char *v6; // r10
  char *v7; // rdi
  _BYTE *v8; // rsi
  signed __int64 v9; // r8
  char *v12; // rdx
  char *v13; // rax

  v5 = *(char **)(a2 + 16);
  v6 = *(char **)(a3 + 16);
  v7 = *(char **)(a2 + 8);
  v8 = *(_BYTE **)(a3 + 8);
  v9 = v5 - v7;
  if ( v5 - v7 > (unsigned __int64)(v6 - v8) )
    return 1;
  if ( v6 - v8 != v9 )
    return 0;
  v12 = *(char **)(a3 + 8);
  if ( v5 != v7 )
  {
    v13 = v7;
    while ( *(_QWORD *)v13 >= *(_QWORD *)v12 )
    {
      if ( *(_QWORD *)v13 > *(_QWORD *)v12 )
        goto LABEL_11;
      v13 += 8;
      v12 += 8;
      if ( v5 == v13 )
        goto LABEL_10;
    }
    return 1;
  }
LABEL_10:
  if ( v6 != v12 )
    return 1;
LABEL_11:
  if ( v9 && memcmp(v7, v8, v5 - v7) )
    return 0;
  return sub_2650CE0(a1, a2, a3);
}

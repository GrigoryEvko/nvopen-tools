// Function: sub_10A4600
// Address: 0x10a4600
//
char __fastcall sub_10A4600(_QWORD **a1, __int64 a2)
{
  _BYTE *v2; // rax
  _BYTE *v3; // rax
  char v4; // dl
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rdx

  if ( !a2 )
    return 0;
  v2 = *(_BYTE **)(a2 - 64);
  if ( *v2 == 58 && (v6 = *((_QWORD *)v2 - 8)) != 0 && (**a1 = v6, (v7 = *((_QWORD *)v2 - 4)) != 0) )
  {
    *a1[1] = v7;
    v3 = *(_BYTE **)(a2 - 32);
    v4 = *v3;
    if ( *v3 == 57 )
    {
      v8 = *((_QWORD *)v3 - 8);
      v9 = *((_QWORD *)v3 - 4);
      v10 = *a1[2];
      return v8 == v10 && *a1[3] == v9 || v10 == v9 && v8 == *a1[3];
    }
  }
  else
  {
    v3 = *(_BYTE **)(a2 - 32);
    v4 = *v3;
  }
  if ( v4 != 58 )
    return 0;
  v11 = *((_QWORD *)v3 - 8);
  if ( !v11 )
    return 0;
  **a1 = v11;
  v12 = *((_QWORD *)v3 - 4);
  if ( !v12 )
    return 0;
  *a1[1] = v12;
  v13 = *(_BYTE **)(a2 - 64);
  if ( *v13 != 57 )
    return 0;
  v14 = *((_QWORD *)v13 - 8);
  v15 = *((_QWORD *)v13 - 4);
  v16 = *a1[2];
  if ( v14 == v16 && *a1[3] == v15 )
    return 1;
  if ( v16 != v15 )
    return 0;
  return *a1[3] == v14;
}

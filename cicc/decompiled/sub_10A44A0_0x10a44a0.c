// Function: sub_10A44A0
// Address: 0x10a44a0
//
bool __fastcall sub_10A44A0(_QWORD **a1, __int64 a2)
{
  _BYTE *v2; // rax
  _BYTE *v3; // rax
  char v4; // dl
  __int64 v6; // rdx
  __int64 v7; // rdx

  if ( !a2 )
    return 0;
  v2 = *(_BYTE **)(a2 - 64);
  if ( *v2 != 68 || (v7 = *((_QWORD *)v2 - 4)) == 0 )
  {
    v3 = *(_BYTE **)(a2 - 32);
    v4 = *v3;
LABEL_4:
    if ( v4 == 68 )
    {
      v6 = *((_QWORD *)v3 - 4);
      if ( v6 )
      {
        **a1 = v6;
        v3 = *(_BYTE **)(a2 - 64);
        if ( *v3 == 69 )
          return *a1[1] == *((_QWORD *)v3 - 4);
      }
    }
    return 0;
  }
  **a1 = v7;
  v3 = *(_BYTE **)(a2 - 32);
  v4 = *v3;
  if ( *v3 != 69 )
    goto LABEL_4;
  return *a1[1] == *((_QWORD *)v3 - 4);
}

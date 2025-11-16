// Function: sub_1181140
// Address: 0x1181140
//
__int64 __fastcall sub_1181140(__int64 *a1, _BYTE *a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx

  v2 = 0;
  if ( *a2 == 82 )
  {
    v4 = *((_QWORD *)a2 - 8);
    v5 = a1[1];
    v6 = *((_QWORD *)a2 - 4);
    if ( v4 == v5 && v6 == a1[2] )
    {
      v2 = 1;
      if ( *a1 )
      {
        v9 = sub_B53900((__int64)a2);
        v10 = *a1;
        *(_DWORD *)v10 = v9;
        *(_BYTE *)(v10 + 4) = BYTE4(v9);
      }
    }
    else
    {
      v2 = 0;
      if ( v5 == v6 && v4 == a1[2] )
      {
        v2 = 1;
        if ( *a1 )
        {
          v7 = sub_B53960((__int64)a2);
          v8 = *a1;
          *(_DWORD *)v8 = v7;
          *(_BYTE *)(v8 + 4) = BYTE4(v7);
        }
      }
    }
  }
  return v2;
}

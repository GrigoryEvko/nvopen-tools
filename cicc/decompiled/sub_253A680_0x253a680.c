// Function: sub_253A680
// Address: 0x253a680
//
__int64 __fastcall sub_253A680(__int64 a1, unsigned __int64 *a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rsi
  bool v5; // dl
  __int64 v6; // rax
  unsigned __int64 v7; // rcx
  bool v8; // cf
  unsigned __int64 v9; // rax
  int v10; // eax
  bool v11; // cf
  bool v12; // zf
  bool v13; // al
  __int64 result; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rax
  int v18; // eax

  v3 = *(_QWORD *)(a1 + 32);
  if ( !v3 )
  {
    v3 = a1 + 24;
    goto LABEL_19;
  }
  v4 = *a2;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v3 + 32);
    v8 = v4 < v7;
    if ( v4 != v7 || (v9 = *(_QWORD *)(v3 + 40), v8 = a2[1] < v9, a2[1] != v9) )
    {
      v5 = v8;
      goto LABEL_4;
    }
    v10 = *(_DWORD *)(v3 + 48);
    if ( *((_DWORD *)a2 + 4) == v10 )
      break;
    v5 = *((_DWORD *)a2 + 4) < v10;
LABEL_4:
    if ( !v5 )
      break;
    v6 = *(_QWORD *)(v3 + 16);
    if ( !v6 )
      goto LABEL_12;
LABEL_6:
    v3 = v6;
  }
  v6 = *(_QWORD *)(v3 + 24);
  v5 = 0;
  if ( v6 )
    goto LABEL_6;
LABEL_12:
  if ( !v5 )
  {
    v11 = v4 < v7;
    v12 = v4 == v7;
    if ( v4 != v7 )
      goto LABEL_14;
LABEL_21:
    v17 = a2[1];
    if ( *(_QWORD *)(v3 + 40) != v17 )
    {
      v13 = *(_QWORD *)(v3 + 40) < v17;
      goto LABEL_15;
    }
    v18 = *((_DWORD *)a2 + 4);
    if ( *(_DWORD *)(v3 + 48) == v18 || *(_DWORD *)(v3 + 48) >= v18 )
      return v3;
    return 0;
  }
LABEL_19:
  result = 0;
  if ( v3 != *(_QWORD *)(a1 + 40) )
  {
    v15 = sub_220EF80(v3);
    v16 = *(_QWORD *)(v15 + 32);
    v3 = v15;
    v11 = *a2 < v16;
    v12 = *a2 == v16;
    if ( *a2 == v16 )
      goto LABEL_21;
LABEL_14:
    v13 = !v11 && !v12;
LABEL_15:
    if ( !v13 )
      return v3;
    return 0;
  }
  return result;
}

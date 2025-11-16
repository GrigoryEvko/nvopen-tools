// Function: sub_17314B0
// Address: 0x17314b0
//
__int64 __fastcall sub_17314B0(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rdx
  int v3; // eax
  __int64 v4; // rdx
  int v5; // eax
  int v7; // eax
  __int64 *v8; // rdx
  __int64 v9; // rdx
  int v10; // eax
  __int64 *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rax

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 48);
  v3 = *(unsigned __int8 *)(v2 + 16);
  if ( (unsigned __int8)v3 > 0x17u )
  {
    v7 = v3 - 24;
  }
  else
  {
    if ( (_BYTE)v3 != 5 )
    {
LABEL_4:
      v4 = *(_QWORD *)(a2 - 24);
      v5 = *(unsigned __int8 *)(v4 + 16);
      goto LABEL_5;
    }
    v7 = *(unsigned __int16 *)(v2 + 18);
  }
  if ( v7 != 37 )
    goto LABEL_4;
  v8 = (*(_BYTE *)(v2 + 23) & 0x40) != 0
     ? *(__int64 **)(v2 - 8)
     : (__int64 *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
  v9 = *v8;
  if ( !v9 )
    goto LABEL_4;
  **a1 = v9;
  v4 = *(_QWORD *)(a2 - 24);
  v5 = *(unsigned __int8 *)(v4 + 16);
  if ( (_BYTE)v5 == 13 )
  {
    *a1[1] = v4;
    return 1;
  }
LABEL_5:
  if ( (unsigned __int8)v5 > 0x17u )
  {
    v10 = v5 - 24;
  }
  else
  {
    if ( (_BYTE)v5 != 5 )
      return 0;
    v10 = *(unsigned __int16 *)(v4 + 18);
  }
  if ( v10 != 37 )
    return 0;
  v11 = (*(_BYTE *)(v4 + 23) & 0x40) != 0
      ? *(__int64 **)(v4 - 8)
      : (__int64 *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
  v12 = *v11;
  if ( !v12 )
    return 0;
  **a1 = v12;
  v13 = *(_QWORD *)(a2 - 48);
  if ( *(_BYTE *)(v13 + 16) != 13 )
    return 0;
  *a1[1] = v13;
  return 1;
}

// Function: sub_163ADA0
// Address: 0x163ada0
//
_BOOL8 __fastcall sub_163ADA0(__int64 a1, const char *a2)
{
  _BYTE *v2; // r8
  _BOOL8 result; // rax
  _BYTE *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  const void *v10; // r13
  size_t v11; // rax

  v2 = *(_BYTE **)(a1 - 16);
  result = 0;
  if ( !*v2 )
  {
    v4 = *(_BYTE **)(a1 - 8);
    if ( !*v4 )
    {
      v5 = sub_161E970((__int64)v2);
      result = 0;
      if ( v6 == 13
        && *(_QWORD *)v5 == 0x46656C69666F7250LL
        && *(_DWORD *)(v5 + 8) == 1634562671
        && *(_BYTE *)(v5 + 12) == 116 )
      {
        v7 = sub_161E970((__int64)v4);
        v9 = v8;
        v10 = (const void *)v7;
        v11 = strlen(a2);
        if ( v11 == v9 && (!v11 || !memcmp(v10, a2, v11)) )
          return 1;
      }
    }
  }
  return result;
}

// Function: sub_1097E30
// Address: 0x1097e30
//
__int64 __fastcall sub_1097E30(__int64 a1, const char *a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6)
{
  __int64 v6; // rax
  size_t v8; // rdx
  __int64 v9; // rsi
  unsigned int v10; // r9d

  v6 = *(_QWORD *)(a1 + 144);
  if ( !*(_BYTE *)(v6 + 22) || (a6 = *(unsigned __int8 *)(a1 + 177), (_BYTE)a6) )
  {
    v8 = *(_QWORD *)(v6 + 56);
    v9 = *(_QWORD *)(v6 + 48);
    if ( v8 != 1 && *(_BYTE *)(v9 + 1) != 35 )
    {
      LOBYTE(v10) = strncmp(a2, (const char *)v9, v8) == 0;
      return v10;
    }
    LOBYTE(a6) = *a2 == *(_BYTE *)v9;
  }
  return a6;
}

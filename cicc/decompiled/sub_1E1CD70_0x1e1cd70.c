// Function: sub_1E1CD70
// Address: 0x1e1cd70
//
__int64 __fastcall sub_1E1CD70(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rsi
  unsigned int v5; // eax
  __int64 v7; // rax
  __int16 v8; // dx
  char v9; // al
  __int64 v10; // rax
  __int64 **v11; // rdx
  __int64 **v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  int v15; // eax
  unsigned int v16; // eax
  char v17[33]; // [rsp+Fh] [rbp-21h] BYREF

  v4 = *(_QWORD *)(a1 + 576);
  v17[0] = 1;
  v5 = sub_1E17B50(a2, v4, v17);
  if ( !(_BYTE)v5 )
  {
    v2 = v5;
    if ( !byte_4FC6740 || !(unsigned __int8)sub_1E1CC30(a2, *(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 264)) )
      return v2;
  }
  v7 = *(_QWORD *)(a2 + 16);
  if ( *(_WORD *)v7 == 1 && (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 8) != 0
    || ((v8 = *(_WORD *)(a2 + 46), (v8 & 4) == 0) && (v8 & 8) != 0
      ? (v9 = sub_1E15D00(a2, 0x10000u, 1))
      : (v9 = WORD1(*(_QWORD *)(v7 + 8)) & 1),
        v9) )
  {
    v10 = *(unsigned __int8 *)(a2 + 49);
    if ( (_BYTE)v10 )
    {
      v11 = *(__int64 ***)(a2 + 56);
      v12 = &v11[v10];
      while ( 1 )
      {
        v13 = **v11;
        if ( (v13 & 4) != 0 )
        {
          v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v14 )
          {
            if ( (*(_DWORD *)(v14 + 8) & 0xFFFFFFFD) == 1 )
              break;
          }
        }
        if ( v12 == ++v11 )
        {
          v15 = *(_DWORD *)(a1 + 1848);
          if ( v15 == 2 )
          {
            LOBYTE(v16) = sub_1E1C9C0(a1, *(_QWORD *)(a2 + 24));
            return v16;
          }
          else
          {
            LOBYTE(v2) = v15 == 0;
          }
          return v2;
        }
      }
    }
  }
  return 1;
}

// Function: sub_3578640
// Address: 0x3578640
//
char __fastcall sub_3578640(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rax
  _BYTE *v5; // rbx
  _BYTE *v6; // r12
  _BYTE *v7; // r14
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned int v10; // esi
  __int64 v11; // rcx
  __int64 v12; // rdx
  int v13; // edi
  _BYTE *v14; // rbx

  v3 = *(_DWORD *)(a2 + 44);
  if ( (v3 & 4) == 0 && (v3 & 8) != 0 )
  {
    LOBYTE(v4) = sub_2E88A90(a2, 512, 1);
    if ( (_BYTE)v4 )
      return v4;
  }
  else
  {
    v4 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 9) & 1LL;
    if ( (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) & 0x200LL) != 0 )
      return v4;
  }
  v5 = *(_BYTE **)(a2 + 32);
  LODWORD(v4) = 5 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  v6 = &v5[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
  if ( v5 != v6 )
  {
    while ( 1 )
    {
      v7 = v5;
      LOBYTE(v4) = sub_2DADC00(v5);
      if ( (_BYTE)v4 )
        break;
      v5 += 40;
      if ( v6 == v5 )
        return v4;
    }
    while ( v6 != v7 )
    {
      LODWORD(v4) = *(_DWORD *)(a1 + 264);
      v10 = *((_DWORD *)v7 + 2);
      v11 = *(_QWORD *)(a1 + 248);
      if ( (_DWORD)v4 )
      {
        v12 = (unsigned int)(v4 - 1);
        LODWORD(v4) = v12 & (37 * v10);
        v13 = *(_DWORD *)(v11 + 4LL * (unsigned int)v4);
        if ( v10 == v13 )
        {
LABEL_13:
          LOBYTE(v4) = sub_3578510(a1, v10, v12, v11, v8, v9);
        }
        else
        {
          v8 = 1;
          while ( v13 != -1 )
          {
            v9 = (unsigned int)(v8 + 1);
            LODWORD(v4) = v12 & (v8 + v4);
            v13 = *(_DWORD *)(v11 + 4LL * (unsigned int)v4);
            if ( v10 == v13 )
              goto LABEL_13;
            v8 = (unsigned int)v9;
          }
        }
      }
      v14 = v7 + 40;
      if ( v7 + 40 == v6 )
        break;
      while ( 1 )
      {
        v7 = v14;
        LOBYTE(v4) = sub_2DADC00(v14);
        if ( (_BYTE)v4 )
          break;
        v14 += 40;
        if ( v6 == v14 )
          return v4;
      }
    }
  }
  return v4;
}

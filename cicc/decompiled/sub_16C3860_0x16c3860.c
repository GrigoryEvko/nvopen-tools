// Function: sub_16C3860
// Address: 0x16c3860
//
__int64 __fastcall sub_16C3860(__int64 a1)
{
  unsigned __int64 v1; // rdx
  unsigned __int64 v2; // rax
  __int64 v3; // r14
  const char *v4; // r13
  size_t v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rdx
  _BYTE *v10; // rax
  int v11; // esi
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // rax
  _BOOL8 v16; // rdx
  bool v17; // zf
  unsigned __int64 v18; // rax
  __int64 v19; // rdx

  v1 = *(_QWORD *)(a1 + 24);
  v2 = v1 + *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 32) = v2;
  if ( v2 == *(_QWORD *)(a1 + 8) )
  {
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  if ( v1 <= 2 )
    goto LABEL_3;
  if ( !sub_16C36C0(**(_BYTE **)(a1 + 16), *(_DWORD *)(a1 + 40))
    || (v10 = *(_BYTE **)(a1 + 16), *v10 != v10[1])
    || sub_16C36C0(v10[2], *(_DWORD *)(a1 + 40)) )
  {
    v2 = *(_QWORD *)(a1 + 32);
LABEL_3:
    if ( !sub_16C36C0(*(_BYTE *)(*(_QWORD *)a1 + v2), *(_DWORD *)(a1 + 40)) )
      goto LABEL_4;
    v11 = *(_DWORD *)(a1 + 40);
    if ( v11 || (v13 = *(_QWORD *)(a1 + 24)) == 0 || *(_BYTE *)(*(_QWORD *)(a1 + 16) + v13 - 1) != 58 )
    {
      v12 = *(_QWORD *)(a1 + 32);
      if ( *(_QWORD *)(a1 + 8) == v12 )
      {
        v19 = *(_QWORD *)(a1 + 32);
      }
      else
      {
        while ( sub_16C36C0(*(_BYTE *)(*(_QWORD *)a1 + v12), v11) )
        {
          v19 = *(_QWORD *)(a1 + 8);
          v12 = *(_QWORD *)(a1 + 32) + 1LL;
          *(_QWORD *)(a1 + 32) = v12;
          if ( v12 == v19 )
            goto LABEL_25;
          v11 = *(_DWORD *)(a1 + 40);
        }
        v3 = *(_QWORD *)(a1 + 32);
        v19 = *(_QWORD *)(a1 + 8);
        if ( v3 != v19 )
          goto LABEL_5;
      }
LABEL_25:
      if ( *(_QWORD *)(a1 + 24) != 1 || **(_BYTE **)(a1 + 16) != 47 )
      {
        *(_QWORD *)(a1 + 24) = 1;
        *(_QWORD *)(a1 + 32) = v19 - 1;
        *(_QWORD *)(a1 + 16) = ".";
        return a1;
      }
LABEL_4:
      v3 = *(_QWORD *)(a1 + 32);
LABEL_5:
      v4 = "/";
      if ( !*(_DWORD *)(a1 + 40) )
        v4 = "\\/";
      v5 = strlen(v4);
      v6 = sub_16D23E0(a1, v4, v5, v3);
      v7 = *(_QWORD *)(a1 + 8);
      v8 = v7;
      if ( *(_QWORD *)(a1 + 32) <= v7 )
        v8 = *(_QWORD *)(a1 + 32);
      if ( v6 < v8 )
        v6 = v8;
      *(_QWORD *)(a1 + 16) = v8 + *(_QWORD *)a1;
      if ( v6 > v7 )
        v6 = v7;
      *(_QWORD *)(a1 + 24) = v6 - v8;
      return a1;
    }
    goto LABEL_30;
  }
  if ( !sub_16C36C0(*(_BYTE *)(*(_QWORD *)a1 + *(_QWORD *)(a1 + 32)), *(_DWORD *)(a1 + 40)) )
    goto LABEL_4;
LABEL_30:
  v14 = *(_QWORD *)(a1 + 32);
  v15 = *(_QWORD *)(a1 + 8);
  v16 = 0;
  v17 = v14 == v15;
  if ( v14 <= v15 )
  {
    v15 = *(_QWORD *)(a1 + 32);
    v16 = !v17;
  }
  v18 = *(_QWORD *)a1 + v15;
  *(_QWORD *)(a1 + 24) = v16;
  *(_QWORD *)(a1 + 16) = v18;
  return a1;
}

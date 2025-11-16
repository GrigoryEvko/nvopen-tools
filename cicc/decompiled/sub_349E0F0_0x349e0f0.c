// Function: sub_349E0F0
// Address: 0x349e0f0
//
bool __fastcall sub_349E0F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // eax
  int v8; // ecx
  int v9; // ecx
  int v10; // r8d
  int v12; // ecx
  __int64 v13; // r11
  __int64 v14; // r10
  unsigned int v15; // ecx
  unsigned int v16; // eax
  __int64 v17; // rbx
  __int64 v18; // r10
  __int64 v19; // r11
  __int64 v20; // r8
  bool v21; // zf

  v5 = a4 - a3;
  v6 = a1 + a4 - a3;
  if ( a2 - a1 > v5 )
    a2 = v6;
  if ( a1 != a2 )
  {
    while ( 1 )
    {
      v7 = *(_DWORD *)a1;
      if ( *(_DWORD *)a1 == 2 )
      {
        v8 = *(_DWORD *)a3;
        if ( *(int *)a3 > 2 )
          return 1;
        if ( v8 == 2 )
        {
          v15 = *(_DWORD *)(a3 + 8);
          v16 = *(_DWORD *)(a1 + 8);
          if ( v15 > v16 )
            return 1;
          v17 = *(_QWORD *)(a3 + 24);
          v18 = *(_QWORD *)(a3 + 16);
          v19 = *(_QWORD *)(a1 + 24);
          v20 = *(_QWORD *)(a1 + 16);
          if ( v15 == v16 && (v18 > v20 || v18 == v20 && v17 > v19) )
            return 1;
        }
        else
        {
LABEL_42:
          if ( v8 == 1 )
            return 0;
          if ( v8 != 2 )
LABEL_55:
            BUG();
          v17 = *(_QWORD *)(a3 + 24);
          v19 = *(_QWORD *)(a1 + 24);
          v18 = *(_QWORD *)(a3 + 16);
          v20 = *(_QWORD *)(a1 + 16);
          v21 = v7 == 2;
          if ( v7 > 2 )
            return 0;
          v16 = *(_DWORD *)(a1 + 8);
          v15 = *(_DWORD *)(a3 + 8);
          if ( !v21 )
            goto LABEL_23;
        }
        if ( v15 < v16 || v15 == v16 && (v20 > v18 || v19 > v17 && v20 == v18) )
          return 0;
        goto LABEL_23;
      }
      if ( v7 <= 2 )
      {
        if ( v7 != 1 )
          goto LABEL_55;
      }
      else if ( v7 != 3 )
      {
        if ( v7 != 4 )
          goto LABEL_55;
        v8 = *(_DWORD *)a3;
        if ( *(int *)a3 > 4 )
          return 1;
        if ( v8 != 4 )
        {
          if ( v8 == 3 )
            return 0;
          goto LABEL_42;
        }
        v9 = *(_DWORD *)(a3 + 8);
        v10 = *(_DWORD *)(a1 + 8);
        if ( v9 > v10 )
          return 1;
        v13 = *(_QWORD *)(a3 + 16);
        v14 = *(_QWORD *)(a1 + 16);
        if ( v9 == v10 && v13 > v14 )
          return 1;
        if ( v10 > v9 )
          return 0;
        if ( v10 == v9 && v13 < v14 )
          return 0;
        goto LABEL_23;
      }
      v12 = *(_DWORD *)a3;
      if ( v7 < *(_DWORD *)a3 || v7 == v12 && *(_QWORD *)(a1 + 8) < *(_QWORD *)(a3 + 8) )
        return 1;
      if ( v12 != 3 )
      {
        if ( v12 > 3 )
        {
          if ( v12 != 4 )
            goto LABEL_55;
          goto LABEL_23;
        }
        if ( v12 != 1 )
        {
          if ( v12 != 2 )
            goto LABEL_55;
          if ( v7 > 2 )
            return 0;
          goto LABEL_23;
        }
      }
      if ( v7 > v12 )
        return 0;
      if ( *(_QWORD *)(a3 + 8) < *(_QWORD *)(a1 + 8) )
        return 0;
LABEL_23:
      a1 += 32;
      a3 += 32;
      if ( a2 == a1 )
        return a4 != a3;
    }
  }
  return a4 != a3;
}

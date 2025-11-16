// Function: sub_2E7A760
// Address: 0x2e7a760
//
void __fastcall sub_2E7A760(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdi
  __int64 v12; // rax

  v2 = a1 + 320;
  v4 = *(_QWORD *)(a1 + 96);
  if ( a1 + 320 != (*(_QWORD *)(a1 + 320) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( !a2 || a2 == *(_QWORD *)(a1 + 328) )
    {
      a2 = *(_QWORD *)(a1 + 328);
      LODWORD(v5) = 0;
      if ( v2 == a2 )
      {
        v9 = *(_QWORD *)(a1 + 104);
        v5 = 0;
        v11 = (v9 - v4) >> 3;
        goto LABEL_13;
      }
    }
    else
    {
      v5 = (unsigned int)(*(_DWORD *)((*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) + 1);
      if ( a2 == v2 )
        goto LABEL_12;
    }
    do
    {
      v6 = *(int *)(a2 + 24);
      if ( (_DWORD)v5 != (_DWORD)v6 )
      {
        if ( (_DWORD)v6 != -1 )
        {
          *(_QWORD *)(v4 + 8 * v6) = 0;
          v4 = *(_QWORD *)(a1 + 96);
        }
        v7 = 8LL * (unsigned int)v5;
        v8 = v7 + v4;
        if ( *(_QWORD *)v8 )
        {
          *(_DWORD *)(*(_QWORD *)v8 + 24LL) = -1;
          v8 = v7 + *(_QWORD *)(a1 + 96);
        }
        *(_QWORD *)v8 = a2;
        *(_DWORD *)(a2 + 24) = v5;
        v4 = *(_QWORD *)(a1 + 96);
      }
      a2 = *(_QWORD *)(a2 + 8);
      v5 = (unsigned int)(v5 + 1);
    }
    while ( v2 != a2 );
LABEL_12:
    v9 = *(_QWORD *)(a1 + 104);
    v10 = (v9 - v4) >> 3;
    v11 = v10;
    if ( v5 > v10 )
    {
      sub_2E7A5B0(a1 + 96, v5 - v10);
LABEL_16:
      ++*(_DWORD *)(a1 + 120);
      return;
    }
LABEL_13:
    if ( v5 < v11 )
    {
      v12 = v4 + 8 * v5;
      if ( v9 != v12 )
        *(_QWORD *)(a1 + 104) = v12;
    }
    goto LABEL_16;
  }
  if ( *(_QWORD *)(a1 + 104) != v4 )
    *(_QWORD *)(a1 + 104) = v4;
}

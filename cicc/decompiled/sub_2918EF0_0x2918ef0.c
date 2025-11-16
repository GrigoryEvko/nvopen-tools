// Function: sub_2918EF0
// Address: 0x2918ef0
//
void __fastcall sub_2918EF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // r14
  unsigned int v5; // ebx
  unsigned __int64 v6; // r15
  unsigned int v7; // r14d
  __int64 v8; // rbx
  __int64 *v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rax
  int v16; // eax
  int v17; // eax
  unsigned int v18; // r14d

  if ( sub_BD2BE0(a2) )
  {
    v12 = *(_QWORD *)(a1 + 376);
    if ( !*(_BYTE *)v12 )
    {
      v13 = *(unsigned int *)(v12 + 320);
      v14 = *(_QWORD *)(a1 + 336);
      if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 324) )
      {
        sub_C8D5F0(v12 + 312, (const void *)(v12 + 328), v13 + 1, 8u, v2, v3);
        v13 = *(unsigned int *)(v12 + 320);
      }
      *(_QWORD *)(*(_QWORD *)(v12 + 312) + 8 * v13) = v14;
      ++*(_DWORD *)(v12 + 320);
    }
  }
  else if ( *(_BYTE *)(a1 + 344) )
  {
    if ( sub_B46A10(a2) )
    {
      if ( !**(_BYTE **)(a1 + 376) )
      {
        v4 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        v5 = *(_DWORD *)(v4 + 32);
        if ( v5 > 0x40 )
        {
          v6 = -1;
          if ( v5 - (unsigned int)sub_C444A0(v4 + 24) <= 0x40 )
            v6 = **(_QWORD **)(v4 + 24);
        }
        else
        {
          v6 = *(_QWORD *)(v4 + 24);
        }
        v7 = *(_DWORD *)(a1 + 360);
        v8 = *(_QWORD *)(a1 + 368);
        v9 = (__int64 *)(a1 + 352);
        if ( v7 > 0x40 )
        {
          v17 = sub_C444A0(a1 + 352);
          v9 = (__int64 *)(a1 + 352);
          v18 = v7 - v17;
          v10 = -1;
          if ( v18 <= 0x40 )
            v10 = **(_QWORD **)(a1 + 352);
        }
        else
        {
          v10 = *(_QWORD *)(a1 + 352);
        }
        v11 = v8 - v10;
        if ( v8 - v10 > v6 )
          v11 = v6;
        sub_2916EE0(a1, a2, v9, v11, 1);
      }
    }
    else if ( (unsigned __int8)sub_B46A50(a2) )
    {
      sub_2916EE0(a1, a2, (__int64 *)(a1 + 352), *(_QWORD *)(a1 + 368), 1);
      sub_3109010(a1, a2);
    }
    else
    {
      v15 = *(_QWORD *)(a2 - 32);
      if ( !v15 || *(_BYTE *)v15 || *(_QWORD *)(v15 + 24) != *(_QWORD *)(a2 + 80) )
        BUG();
      v16 = *(_DWORD *)(v15 + 36);
      if ( v16 == 171 )
      {
        *(_QWORD *)(a1 + 16) = a2;
      }
      else if ( (unsigned int)(v16 - 210) > 1 )
      {
        sub_2918CF0((_QWORD *)a1, (unsigned __int8 *)a2);
      }
    }
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a2;
  }
}

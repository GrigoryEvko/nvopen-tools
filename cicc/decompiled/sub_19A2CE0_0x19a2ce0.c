// Function: sub_19A2CE0
// Address: 0x19a2ce0
//
bool __fastcall sub_19A2CE0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // r15
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v11; // r8
  __int64 v12; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  _QWORD *v16; // rdx
  _QWORD *v17; // [rsp+8h] [rbp-48h]
  __int64 v18; // [rsp+18h] [rbp-38h]

  v3 = *a1;
  if ( *(_BYTE *)(*a1 + 16LL) == 77 )
  {
    v4 = 0;
    v5 = a2 + 56;
    v6 = 8LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
    if ( (*(_DWORD *)(v3 + 20) & 0xFFFFFFF) == 0 )
      return 1;
    while ( 1 )
    {
      if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
      {
        v7 = *(_QWORD *)(v3 - 8);
        if ( a1[1] != *(_QWORD *)(v7 + 3 * v4) )
          goto LABEL_5;
      }
      else
      {
        v7 = v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
        if ( a1[1] != *(_QWORD *)(v7 + 3 * v4) )
          goto LABEL_5;
      }
      v8 = *(_QWORD **)(a2 + 72);
      v9 = *(_QWORD *)(v4 + v7 + 24LL * *(unsigned int *)(v3 + 56) + 8);
      v10 = *(_QWORD **)(a2 + 64);
      if ( v8 == v10 )
      {
        v11 = &v10[*(unsigned int *)(a2 + 84)];
        if ( v10 == v11 )
        {
          v16 = *(_QWORD **)(a2 + 64);
        }
        else
        {
          do
          {
            if ( v9 == *v10 )
              break;
            ++v10;
          }
          while ( v11 != v10 );
          v16 = v11;
        }
      }
      else
      {
        v18 = v5;
        v17 = &v8[*(unsigned int *)(a2 + 80)];
        v10 = sub_16CC9F0(v5, v9);
        v5 = v18;
        v11 = v17;
        if ( v9 == *v10 )
        {
          v14 = *(_QWORD *)(a2 + 72);
          if ( v14 == *(_QWORD *)(a2 + 64) )
            v15 = *(unsigned int *)(a2 + 84);
          else
            v15 = *(unsigned int *)(a2 + 80);
          v16 = (_QWORD *)(v14 + 8 * v15);
        }
        else
        {
          v12 = *(_QWORD *)(a2 + 72);
          if ( v12 != *(_QWORD *)(a2 + 64) )
          {
            v10 = (_QWORD *)(v12 + 8LL * *(unsigned int *)(a2 + 80));
            goto LABEL_12;
          }
          v10 = (_QWORD *)(v12 + 8LL * *(unsigned int *)(a2 + 84));
          v16 = v10;
        }
      }
      for ( ; v16 != v10; ++v10 )
      {
        if ( *v10 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
LABEL_12:
      if ( v10 != v11 )
        return 0;
LABEL_5:
      v4 += 8;
      if ( v6 == v4 )
        return 1;
    }
  }
  return !sub_1377F70(a2 + 56, *(_QWORD *)(v3 + 40));
}

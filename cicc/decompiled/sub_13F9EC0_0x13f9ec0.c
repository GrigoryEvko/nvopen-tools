// Function: sub_13F9EC0
// Address: 0x13f9ec0
//
_QWORD *__fastcall sub_13F9EC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _QWORD *result; // rax
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // r13
  unsigned int v9; // r12d
  _QWORD *v10; // r14
  _QWORD *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  _QWORD *v19; // rdx
  _QWORD *v20; // [rsp+8h] [rbp-58h]
  _QWORD *v21; // [rsp+10h] [rbp-50h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  int v23; // [rsp+24h] [rbp-3Ch]
  __int64 v24; // [rsp+28h] [rbp-38h]
  __int64 v25; // [rsp+28h] [rbp-38h]

  v3 = a1 + 56;
  result = *(_QWORD **)(v3 - 16);
  v22 = v3;
  v20 = result;
  v21 = *(_QWORD **)(v3 - 24);
  if ( v21 != result )
  {
    while ( 1 )
    {
      v6 = *v21;
      v7 = sub_157EBA0(*v21);
      if ( v7 )
      {
        v23 = sub_15F4D60(v7);
        v8 = sub_157EBA0(v6);
        if ( v23 )
          break;
      }
LABEL_23:
      result = ++v21;
      if ( v20 == v21 )
        return result;
    }
    v9 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = sub_15F4DF0(v8, v9);
        v15 = *(_QWORD **)(a1 + 72);
        v12 = v14;
        v11 = *(_QWORD **)(a1 + 64);
        if ( v15 == v11 )
          break;
        v24 = v12;
        v10 = &v15[*(unsigned int *)(a1 + 80)];
        v11 = (_QWORD *)sub_16CC9F0(v22, v12);
        v12 = v24;
        if ( v24 == *v11 )
        {
          v17 = *(_QWORD *)(a1 + 72);
          if ( v17 == *(_QWORD *)(a1 + 64) )
            v18 = *(unsigned int *)(a1 + 84);
          else
            v18 = *(unsigned int *)(a1 + 80);
          v19 = (_QWORD *)(v17 + 8 * v18);
          goto LABEL_18;
        }
        v13 = *(_QWORD *)(a1 + 72);
        if ( v13 == *(_QWORD *)(a1 + 64) )
        {
          v11 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a1 + 84));
          v19 = v11;
          goto LABEL_18;
        }
        v11 = (_QWORD *)(v13 + 8LL * *(unsigned int *)(a1 + 80));
LABEL_8:
        if ( v10 == v11 )
          goto LABEL_20;
LABEL_9:
        if ( v23 == ++v9 )
          goto LABEL_23;
      }
      v10 = &v11[*(unsigned int *)(a1 + 84)];
      if ( v11 == v10 )
      {
        v19 = *(_QWORD **)(a1 + 64);
      }
      else
      {
        do
        {
          if ( v12 == *v11 )
            break;
          ++v11;
        }
        while ( v10 != v11 );
        v19 = v10;
      }
LABEL_18:
      while ( v19 != v11 )
      {
        if ( *v11 < 0xFFFFFFFFFFFFFFFELL )
          goto LABEL_8;
        ++v11;
      }
      if ( v10 != v11 )
        goto LABEL_9;
LABEL_20:
      v16 = *(unsigned int *)(a2 + 8);
      if ( (unsigned int)v16 >= *(_DWORD *)(a2 + 12) )
      {
        v25 = v12;
        sub_16CD150(a2, a2 + 16, 0, 8);
        v16 = *(unsigned int *)(a2 + 8);
        v12 = v25;
      }
      ++v9;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v16) = v12;
      ++*(_DWORD *)(a2 + 8);
      if ( v23 == v9 )
        goto LABEL_23;
    }
  }
  return result;
}

// Function: sub_2997F70
// Address: 0x2997f70
//
_QWORD *__fastcall sub_2997F70(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r9
  _QWORD *v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rax
  _QWORD *v7; // r13
  __int64 v9; // rax
  _QWORD *v10; // r14
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r10
  __int64 v15; // rsi
  unsigned __int8 *v16; // r15
  unsigned __int8 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rbx
  unsigned __int8 *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rdx
  bool v24; // cl
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // [rsp+8h] [rbp-38h]

  v2 = (_QWORD *)(*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v2 == (_QWORD *)(a2 + 48) )
    goto LABEL_47;
  if ( !v2 )
    BUG();
  v3 = *(_QWORD **)(a2 + 56);
  if ( (unsigned int)*((unsigned __int8 *)v2 - 24) - 30 > 0xA )
  {
LABEL_47:
    if ( *(_QWORD *)(a2 + 56) )
LABEL_44:
      BUG();
    return 0;
  }
  if ( v3 )
  {
    v4 = v3 - 3;
    if ( v3 - 3 != v2 - 3 )
      goto LABEL_8;
    return 0;
  }
  v4 = 0;
LABEL_8:
  while ( 1 )
  {
    if ( *((_BYTE *)v2 - 24) == 85 )
    {
      v5 = *(v2 - 7);
      v6 = 0;
      if ( v5 )
      {
        if ( !*(_BYTE *)v5 )
        {
          v6 = 0;
          if ( *(_QWORD *)(v5 + 24) == v2[7] )
            v6 = *(v2 - 7);
        }
      }
      if ( *(_QWORD *)a1 == v6 )
        break;
    }
    if ( v2 == v3 )
      return 0;
    v2 = (_QWORD *)(*v2 & 0xFFFFFFFFFFFFFFF8LL);
    if ( !v2 )
      goto LABEL_44;
  }
  if ( (*((_WORD *)v2 - 11) & 3u) - 1 > 1 )
    return 0;
  v9 = *(_QWORD *)(v6 + 80);
  v10 = v2 - 3;
  v7 = v2 - 3;
  if ( v9 )
  {
    if ( a2 == v9 - 24 && v10 == (_QWORD *)sub_2997170((__int64)(v4 + 3)) )
    {
      v12 = sub_2997170(*(_QWORD *)(v11 + 8));
      if ( v13 )
      {
        if ( v12 == v14 && !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == v2[7] )
        {
          v15 = v13;
          if ( !(unsigned __int8)sub_DF9C30(*(__int64 **)(a1 + 8), (_BYTE *)v13) )
          {
            v16 = (unsigned __int8 *)&v10[-4 * (*((_DWORD *)v2 - 5) & 0x7FFFFFF)];
            v17 = sub_24E54B0((unsigned __int8 *)v2 - 24);
            v20 = *(_QWORD *)a1;
            v21 = v17;
            if ( (*(_BYTE *)(*(_QWORD *)a1 + 2LL) & 1) != 0 )
            {
              sub_B2C6D0(*(_QWORD *)a1, v15, v18, v19);
              v22 = *(_QWORD *)(v20 + 96);
              v20 = *(_QWORD *)a1;
              if ( (*(_BYTE *)(*(_QWORD *)a1 + 2LL) & 1) != 0 )
              {
                v27 = v22;
                sub_B2C6D0(*(_QWORD *)a1, v15, v25, v26);
                v22 = v27;
              }
            }
            else
            {
              v22 = *(_QWORD *)(v20 + 96);
            }
            v23 = *(_QWORD *)(v20 + 96) + 40LL * *(_QWORD *)(v20 + 104);
            v24 = v23 == v22;
            if ( v23 == v22 )
            {
              if ( v21 != v16 )
                return v7;
LABEL_34:
              if ( !v24 )
                return v7;
              return 0;
            }
            if ( v21 != v16 )
            {
              while ( *(_QWORD *)v16 == v22 )
              {
                v16 += 32;
                v22 += 40;
                v24 = v22 == v23;
                if ( v21 == v16 )
                  goto LABEL_34;
                if ( v22 == v23 )
                  return v7;
              }
            }
          }
        }
      }
    }
  }
  return v7;
}

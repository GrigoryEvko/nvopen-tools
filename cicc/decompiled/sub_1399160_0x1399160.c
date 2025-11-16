// Function: sub_1399160
// Address: 0x1399160
//
__int64 __fastcall sub_1399160(_QWORD *a1, unsigned __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // rbx
  __int64 v11; // rdx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r13
  unsigned __int64 *v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // [rsp+10h] [rbp-60h]
  unsigned __int64 v20; // [rsp+20h] [rbp-50h]
  __int64 v21; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v22[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = sub_1399010(a1, a2);
  if ( (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1 || (unsigned __int8)sub_15E3650(a2, 0) )
  {
    v21 = v3;
    v22[0] = 0;
    v4 = a1[7];
    v5 = *(_QWORD *)(v4 + 16);
    if ( v5 == *(_QWORD *)(v4 + 24) )
    {
      sub_1398B10((char **)(v4 + 8), (char *)v5, v22, &v21);
      v6 = v21;
    }
    else
    {
      v6 = v3;
      if ( v5 )
      {
        *(_QWORD *)v5 = 6;
        *(_QWORD *)(v5 + 8) = 0;
        *(_QWORD *)(v5 + 16) = 0;
        v6 = v21;
        *(_QWORD *)(v5 + 24) = v21;
        v5 = *(_QWORD *)(v4 + 16);
      }
      *(_QWORD *)(v4 + 16) = v5 + 32;
    }
    ++*(_DWORD *)(v6 + 32);
  }
  if ( (unsigned __int8)sub_15E4F60(a2) && (*(_BYTE *)(a2 + 33) & 0x20) == 0 )
  {
    v22[0] = 0;
    v17 = a1[8];
    v21 = v17;
    v18 = *(_QWORD *)(v3 + 16);
    if ( v18 == *(_QWORD *)(v3 + 24) )
    {
      sub_1398B10((char **)(v3 + 8), (char *)v18, v22, &v21);
      v17 = v21;
    }
    else
    {
      if ( v18 )
      {
        *(_QWORD *)v18 = 6;
        *(_QWORD *)(v18 + 8) = 0;
        *(_QWORD *)(v18 + 16) = 0;
        v17 = v21;
        *(_QWORD *)(v18 + 24) = v21;
        v18 = *(_QWORD *)(v3 + 16);
      }
      *(_QWORD *)(v3 + 16) = v18 + 32;
    }
    ++*(_DWORD *)(v17 + 32);
  }
  v7 = *(_QWORD *)(a2 + 80);
  result = a2 + 72;
  v19 = a2 + 72;
  if ( v7 != a2 + 72 )
  {
    while ( 1 )
    {
      if ( !v7 )
        BUG();
      v9 = *(_QWORD *)(v7 + 24);
      v10 = v7 + 16;
      if ( v7 + 16 != v9 )
        break;
LABEL_33:
      v7 = *(_QWORD *)(v7 + 8);
      if ( v19 == v7 )
        return result;
    }
    while ( 1 )
    {
      if ( !v9 )
        BUG();
      result = *(unsigned __int8 *)(v9 - 8);
      v11 = v9 - 24;
      if ( (unsigned __int8)result <= 0x17u )
        goto LABEL_15;
      if ( (_BYTE)result == 78 )
      {
        v12 = v11 | 4;
      }
      else
      {
        if ( (_BYTE)result != 29 )
          goto LABEL_15;
        v12 = v11 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v13 = v12 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_15;
      v14 = (unsigned __int64 *)(v13 - 72);
      if ( (v12 & 4) != 0 )
        v14 = (unsigned __int64 *)(v13 - 24);
      v15 = *v14;
      if ( *(_BYTE *)(*v14 + 16)
        || (v20 = v12, result = sub_15E1830(*(unsigned int *)(v15 + 36)), v12 = v20, !(_BYTE)result) )
      {
        v22[0] = v13;
        result = a1[8];
        v21 = result;
        v16 = *(_QWORD *)(v3 + 16);
        if ( v16 == *(_QWORD *)(v3 + 24) )
        {
LABEL_40:
          sub_1398B10((char **)(v3 + 8), (char *)v16, v22, &v21);
          result = v21;
          goto LABEL_32;
        }
        if ( v16 )
        {
          *(_QWORD *)v16 = 6;
          *(_QWORD *)(v16 + 8) = 0;
          *(_QWORD *)(v16 + 16) = v13;
          if ( (v12 & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL )
LABEL_29:
            sub_164C220(v16);
LABEL_30:
          result = v21;
          *(_QWORD *)(v16 + 24) = v21;
          v16 = *(_QWORD *)(v3 + 16);
        }
LABEL_31:
        *(_QWORD *)(v3 + 16) = v16 + 32;
LABEL_32:
        ++*(_DWORD *)(result + 32);
        v9 = *(_QWORD *)(v9 + 8);
        if ( v10 == v9 )
          goto LABEL_33;
      }
      else
      {
        if ( (*(_BYTE *)(v15 + 33) & 0x20) == 0 )
        {
          result = sub_1399010(a1, v15);
          v22[0] = v13;
          v21 = result;
          v16 = *(_QWORD *)(v3 + 16);
          if ( v16 == *(_QWORD *)(v3 + 24) )
            goto LABEL_40;
          if ( v16 )
          {
            *(_QWORD *)v16 = 6;
            *(_QWORD *)(v16 + 8) = 0;
            *(_QWORD *)(v16 + 16) = v13;
            if ( (v20 & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL )
              goto LABEL_29;
            goto LABEL_30;
          }
          goto LABEL_31;
        }
LABEL_15:
        v9 = *(_QWORD *)(v9 + 8);
        if ( v10 == v9 )
          goto LABEL_33;
      }
    }
  }
  return result;
}

// Function: sub_2B141D0
// Address: 0x2b141d0
//
__int64 *__fastcall sub_2B141D0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 *v5; // r13
  __int64 v6; // r15
  __int64 *v7; // r12
  __int64 v8; // r15
  __int64 v9; // r15
  _BYTE *v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rdx
  _BYTE *v16; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rcx
  _BYTE *v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // rcx
  _BYTE *v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rdx
  _BYTE *v35; // rdi

  v2 = ((char *)a2 - (char *)a1) >> 5;
  v3 = a1;
  v4 = a2 - a1;
  if ( v2 > 0 )
  {
    v5 = &a1[4 * v2];
    while ( 1 )
    {
      v10 = (_BYTE *)*v3;
      if ( *(_BYTE *)*v3 != 86 )
        goto LABEL_3;
      v11 = *((_QWORD *)v10 + 1);
      if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
        v11 = **(_QWORD **)(v11 + 16);
      if ( !sub_BCAC40(v11, 1) )
        goto LABEL_71;
      if ( *v10 == 57 )
        return v3;
      v12 = *((_QWORD *)v10 + 1);
      if ( *v10 == 86 && *(_QWORD *)(*((_QWORD *)v10 - 12) + 8LL) == v12 && **((_BYTE **)v10 - 4) <= 0x15u )
        break;
LABEL_14:
      if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
        v12 = **(_QWORD **)(v12 + 16);
      if ( sub_BCAC40(v12, 1) )
      {
        if ( *v10 == 58 )
          return v3;
        if ( *v10 == 86 )
        {
          v15 = *((_QWORD *)v10 + 1);
          if ( *(_QWORD *)(*((_QWORD *)v10 - 12) + 8LL) == v15 )
          {
            v16 = (_BYTE *)*((_QWORD *)v10 - 8);
            if ( *v16 <= 0x15u && sub_AD7A80(v16, 1, v15, v13, v14) )
              return v3;
          }
        }
      }
LABEL_3:
      v6 = v3[1];
      v7 = v3 + 1;
      if ( *(_BYTE *)v6 != 86 )
        goto LABEL_4;
      v18 = *(_QWORD *)(v6 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
        v18 = **(_QWORD **)(v18 + 16);
      if ( !sub_BCAC40(v18, 1) )
        goto LABEL_74;
      if ( *(_BYTE *)v6 == 57 )
        return v7;
      v19 = *(_QWORD *)(v6 + 8);
      if ( *(_BYTE *)v6 == 86 && *(_QWORD *)(*(_QWORD *)(v6 - 96) + 8LL) == v19 && **(_BYTE **)(v6 - 32) <= 0x15u )
      {
        if ( sub_AC30F0(*(_QWORD *)(v6 - 32)) )
          return v7;
LABEL_74:
        v19 = *(_QWORD *)(v6 + 8);
      }
      if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
        v19 = **(_QWORD **)(v19 + 16);
      if ( sub_BCAC40(v19, 1) )
      {
        if ( *(_BYTE *)v6 == 58 )
          return v7;
        if ( *(_BYTE *)v6 == 86 )
        {
          v22 = *(_QWORD *)(v6 + 8);
          if ( *(_QWORD *)(*(_QWORD *)(v6 - 96) + 8LL) == v22 )
          {
            v23 = *(_BYTE **)(v6 - 64);
            if ( *v23 <= 0x15u && sub_AD7A80(v23, 1, v20, v22, v21) )
              return v7;
          }
        }
      }
LABEL_4:
      v8 = v3[2];
      v7 = v3 + 2;
      if ( *(_BYTE *)v8 != 86 )
        goto LABEL_5;
      v24 = *(_QWORD *)(v8 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 <= 1 )
        v24 = **(_QWORD **)(v24 + 16);
      if ( !sub_BCAC40(v24, 1) )
        goto LABEL_77;
      if ( *(_BYTE *)v8 == 57 )
        return v7;
      v25 = *(_QWORD *)(v8 + 8);
      if ( *(_BYTE *)v8 == 86 && *(_QWORD *)(*(_QWORD *)(v8 - 96) + 8LL) == v25 && **(_BYTE **)(v8 - 32) <= 0x15u )
      {
        if ( sub_AC30F0(*(_QWORD *)(v8 - 32)) )
          return v7;
LABEL_77:
        v25 = *(_QWORD *)(v8 + 8);
      }
      if ( (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17 <= 1 )
        v25 = **(_QWORD **)(v25 + 16);
      if ( sub_BCAC40(v25, 1) )
      {
        if ( *(_BYTE *)v8 == 58 )
          return v7;
        if ( *(_BYTE *)v8 == 86 )
        {
          v28 = *(_QWORD *)(v8 + 8);
          if ( *(_QWORD *)(*(_QWORD *)(v8 - 96) + 8LL) == v28 )
          {
            v29 = *(_BYTE **)(v8 - 64);
            if ( *v29 <= 0x15u && sub_AD7A80(v29, 1, v26, v28, v27) )
              return v7;
          }
        }
      }
LABEL_5:
      v9 = v3[3];
      v7 = v3 + 3;
      if ( *(_BYTE *)v9 == 86 )
      {
        v30 = *(_QWORD *)(v9 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v30 + 8) - 17 <= 1 )
          v30 = **(_QWORD **)(v30 + 16);
        if ( !sub_BCAC40(v30, 1) )
          goto LABEL_80;
        if ( *(_BYTE *)v9 == 57 )
          return v7;
        v31 = *(_QWORD *)(v9 + 8);
        if ( *(_BYTE *)v9 == 86 && v31 == *(_QWORD *)(*(_QWORD *)(v9 - 96) + 8LL) && **(_BYTE **)(v9 - 32) <= 0x15u )
        {
          if ( sub_AC30F0(*(_QWORD *)(v9 - 32)) )
            return v7;
LABEL_80:
          v31 = *(_QWORD *)(v9 + 8);
        }
        if ( (unsigned int)*(unsigned __int8 *)(v31 + 8) - 17 <= 1 )
          v31 = **(_QWORD **)(v31 + 16);
        if ( sub_BCAC40(v31, 1) )
        {
          if ( *(_BYTE *)v9 == 58 )
            return v7;
          if ( *(_BYTE *)v9 == 86 )
          {
            v34 = *(_QWORD *)(v9 + 8);
            if ( *(_QWORD *)(*(_QWORD *)(v9 - 96) + 8LL) == v34 )
            {
              v35 = *(_BYTE **)(v9 - 64);
              if ( *v35 <= 0x15u && sub_AD7A80(v35, 1, v34, v32, v33) )
                return v7;
            }
          }
        }
      }
      v3 += 4;
      if ( v5 == v3 )
      {
        v4 = a2 - v3;
        goto LABEL_82;
      }
    }
    if ( sub_AC30F0(*((_QWORD *)v10 - 4)) )
      return v3;
LABEL_71:
    v12 = *((_QWORD *)v10 + 1);
    goto LABEL_14;
  }
LABEL_82:
  if ( v4 != 2 )
  {
    if ( v4 != 3 )
    {
      v7 = a2;
      if ( v4 != 1 )
        return v7;
      goto LABEL_85;
    }
    v7 = v3;
    if ( sub_2B0EFA0(*v3) )
      return v7;
    ++v3;
  }
  v7 = v3;
  if ( sub_2B0EFA0(*v3) )
    return v7;
  ++v3;
LABEL_85:
  if ( !sub_2B0EFA0(*v3) )
    return a2;
  return v3;
}

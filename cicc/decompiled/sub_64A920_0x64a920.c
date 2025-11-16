// Function: sub_64A920
// Address: 0x64a920
//
char __fastcall sub_64A920(_QWORD *a1, __int64 a2)
{
  _QWORD *j; // rax
  __int64 v3; // rbx
  __int64 v4; // rdx
  const char *v5; // r13
  __int64 m; // rbx
  __int64 v7; // r15
  _QWORD *v8; // r14
  unsigned int v9; // eax
  bool v10; // cf
  bool v11; // zf
  __int64 i; // rax
  const char *v13; // rsi
  __int64 k; // rax
  _QWORD *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rcx
  const char *v18; // rdi
  const char *v19; // rsi
  __int64 v20; // r9
  __int64 v21; // rcx
  const char *v22; // rdi
  const char *v23; // rsi
  __int64 v24; // rcx
  const char *v25; // rdi
  const char *v26; // rsi
  __int64 v27; // rcx
  const char *v28; // rdi
  const char *v29; // rsi
  __int64 v30; // rcx
  const char *v31; // rdi
  const char *v32; // rsi
  char v33; // dl
  bool v34; // cf
  bool v35; // zf
  __int64 v36; // rcx
  const char *v37; // rdi
  const char *v38; // rsi
  bool v39; // cf
  bool v40; // zf
  __int64 v41; // rcx
  const char *v42; // rdi
  const char *v43; // rsi
  __int64 v44; // rcx
  const char *v45; // rdi
  const char *v46; // rsi
  __int64 v47; // rcx
  char *v48; // rdi
  char *v49; // rsi

  j = (_QWORD *)a1[5];
  if ( !j )
    return (char)j;
  v3 = a2;
  if ( *((_BYTE *)j + 28) != 3 )
    goto LABEL_3;
  a2 = unk_4D049A0;
  if ( (unsigned int)sub_879C10(*a1, unk_4D049A0) )
  {
    j = *(_QWORD **)(v3 + 8);
    if ( *(_BYTE *)j != 95 )
      return (char)j;
    if ( *((_BYTE *)j + 1) != 95 )
      return (char)j;
    if ( *((_BYTE *)j + 2) != 100 )
      return (char)j;
    v11 = strcmp((const char *)j + 2, "device_construct_at") == 0;
    LOBYTE(j) = !v11;
    if ( !v11 )
      return (char)j;
    j = (_QWORD *)a1[30];
    if ( !j || *((_BYTE *)j + 8) )
      return (char)j;
    for ( i = a1[19]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    goto LABEL_20;
  }
  j = (_QWORD *)a1[5];
  if ( !j )
    return (char)j;
  if ( *((_BYTE *)j + 28) == 3 )
  {
    a2 = unk_4D049B8;
    if ( !(unsigned int)sub_879C10(*a1, unk_4D049B8) )
    {
      j = &qword_4D049B0;
      v4 = qword_4D049B0;
      if ( !qword_4D049B0 )
        return (char)j;
      j = (_QWORD *)a1[5];
      if ( !j )
        return (char)j;
      goto LABEL_4;
    }
    v13 = *(const char **)(v3 + 8);
    LOBYTE(j) = *v13;
    if ( *v13 != 99 )
    {
      if ( (_BYTE)j == 105 )
      {
        v11 = strcmp(v13, "is_constant_evaluated") == 0;
        LOBYTE(j) = !v11;
        if ( v11 )
        {
          for ( j = (_QWORD *)a1[19]; *((_BYTE *)j + 140) == 12; j = (_QWORD *)j[20] )
            ;
          if ( !*(_QWORD *)j[21] )
          {
            LODWORD(j) = sub_8D29A0(j[20]);
            if ( (_DWORD)j )
              LOBYTE(j) = sub_77FA10(1);
          }
        }
      }
      else if ( (_BYTE)j == 95 )
      {
        v11 = strcmp(v13, "__report_constexpr_value") == 0;
        LOBYTE(j) = !v11;
        if ( v11 )
        {
          for ( k = a1[19]; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
            ;
          v15 = **(_QWORD ***)(k + 168);
          LODWORD(j) = sub_8D2600(*(_QWORD *)(k + 160));
          if ( (_DWORD)j && v15 )
          {
            if ( (unsigned int)sub_8D2780(v15[1]) && !*v15
              || (LODWORD(j) = sub_8D2E30(v15[1]), (_DWORD)j)
              && (v16 = sub_8D46C0(v15[1]), LODWORD(j) = sub_8D29E0(v16), (_DWORD)j)
              && (!*v15
               || (LODWORD(j) = sub_8D2780(*(_QWORD *)(*v15 + 8LL)), (_DWORD)j)
               && (LODWORD(j) = sub_8D27E0(*(_QWORD *)(*v15 + 8LL)), (_DWORD)j)
               && (j = (_QWORD *)*v15, !*(_QWORD *)*v15)) )
            {
              LOBYTE(j) = sub_77FA10(6);
            }
          }
        }
      }
      return (char)j;
    }
    v11 = strcmp(v13, "construct_at") == 0;
    LOBYTE(j) = !v11;
    if ( !v11 )
      return (char)j;
    j = (_QWORD *)a1[30];
    if ( !j || *((_BYTE *)j + 8) )
      return (char)j;
    for ( i = a1[19]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
LABEL_20:
    j = **(_QWORD ***)(i + 168);
    if ( j )
    {
      LODWORD(j) = sub_8D2E30(j[1]);
      if ( (_DWORD)j )
        LOBYTE(j) = sub_77FA10(4);
    }
    return (char)j;
  }
LABEL_3:
  v4 = qword_4D049B0;
  if ( !qword_4D049B0 )
    return (char)j;
LABEL_4:
  if ( *((_BYTE *)j + 28) == 3 && j[4] == *(_QWORD *)(v4 + 88) )
  {
    v5 = *(const char **)(v3 + 8);
    for ( m = a1[19]; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
      ;
    v7 = a1[30];
    v8 = **(_QWORD ***)(m + 168);
    v9 = *(unsigned __int8 *)v5;
    v10 = v9 < 0x5F;
    LODWORD(j) = v9 - 95;
    v11 = (_DWORD)j == 0;
    switch ( (char)j )
    {
      case 0:
        v41 = 15;
        v42 = "__define_class";
        v43 = v5;
        do
        {
          if ( !v41 )
            break;
          v10 = *v43 < (unsigned int)*v42;
          v11 = *v43++ == *v42++;
          --v41;
        }
        while ( v11 );
        LOBYTE(j) = (!v10 && !v11) - v10;
        if ( v7 == 0 && v8 != 0 && (!v10 && !v11) == v10 )
        {
          j = (_QWORD *)*v8;
          if ( *v8 )
          {
            j = (_QWORD *)*j;
            if ( j )
            {
              if ( !*j )
              {
                LODWORD(j) = sub_8D2630(v8[1], v43);
                if ( (_DWORD)j )
                {
                  LODWORD(j) = sub_8D2780(*(_QWORD *)(*v8 + 8LL));
                  if ( (_DWORD)j )
                  {
                    LODWORD(j) = sub_8D2E30(*(_QWORD *)(*(_QWORD *)*v8 + 8LL));
                    if ( (_DWORD)j )
                    {
                      LODWORD(j) = sub_8D2600(*(_QWORD *)(m + 160));
                      if ( (_DWORD)j )
                        LOBYTE(j) = sub_77FA10(61);
                    }
                  }
                }
              }
            }
          }
        }
        return (char)j;
      case 2:
        if ( !v7 )
        {
          if ( v8 )
          {
            if ( !*v8 )
            {
              LODWORD(j) = sub_8D2630(v8[1], a2);
              if ( (_DWORD)j )
              {
                LODWORD(j) = sub_8D3A00(*(_QWORD *)(m + 160));
                if ( (_DWORD)j )
                {
                  LODWORD(j) = strcmp(v5, "alignment_of");
                  if ( !(_DWORD)j )
                    LOBYTE(j) = sub_77FA10(60);
                }
              }
            }
          }
        }
        return (char)j;
      case 3:
        if ( !v7 )
        {
          if ( v8 )
          {
            if ( !*v8 )
            {
              LODWORD(j) = sub_8D2630(v8[1], a2);
              if ( (_DWORD)j )
              {
                if ( (unsigned int)sub_8D3A00(*(_QWORD *)(m + 160)) )
                {
                  if ( !strcmp(v5, "bit_offset_of") )
                  {
                    LOBYTE(j) = sub_77FA10(59);
                  }
                  else
                  {
                    LODWORD(j) = strcmp(v5, "bit_size_of");
                    if ( !(_DWORD)j )
                      LOBYTE(j) = sub_77FA10(58);
                  }
                }
                else
                {
                  LODWORD(j) = sub_641850(*(_QWORD *)(m + 160));
                  if ( (_DWORD)j )
                  {
                    LODWORD(j) = strcmp(v5, "bases__impl");
                    if ( !(_DWORD)j )
                      LOBYTE(j) = sub_77FA10(12);
                  }
                }
              }
            }
          }
        }
        return (char)j;
      case 5:
        if ( !v7 )
        {
          if ( v8 )
          {
            if ( !*v8 )
            {
              LODWORD(j) = sub_8D2630(v8[1], a2);
              if ( (_DWORD)j )
              {
                LODWORD(j) = sub_8D2630(*(_QWORD *)(m + 160), a2);
                if ( (_DWORD)j )
                {
                  LODWORD(j) = strcmp(v5, "dealias");
                  if ( !(_DWORD)j )
                    LOBYTE(j) = sub_77FA10(55);
                }
              }
            }
          }
        }
        return (char)j;
      case 6:
        if ( !v7 )
        {
          if ( v8 )
          {
            if ( !*v8 )
            {
              LODWORD(j) = sub_8D2630(v8[1], a2);
              if ( (_DWORD)j )
              {
                LODWORD(j) = sub_641850(*(_QWORD *)(m + 160));
                if ( (_DWORD)j )
                {
                  LODWORD(j) = strcmp(v5, "enumerators__impl");
                  if ( !(_DWORD)j )
                    LOBYTE(j) = sub_77FA10(14);
                }
              }
            }
          }
        }
        return (char)j;
      case 9:
        if ( !v7 )
        {
          if ( v8 )
          {
            if ( !*v8 )
            {
              LODWORD(j) = sub_8D2630(v8[1], a2);
              if ( (_DWORD)j )
              {
                LODWORD(j) = sub_8D29A0(*(_QWORD *)(m + 160));
                if ( (_DWORD)j )
                {
                  if ( !strcmp(v5, "has_template_arguments") )
                  {
                    LOBYTE(j) = sub_77FA10(50);
                  }
                  else if ( !strcmp(v5, "has_static_storage_duration") )
                  {
                    LOBYTE(j) = sub_77FA10(48);
                  }
                  else
                  {
                    LODWORD(j) = strcmp(v5, "has_internal_linkage");
                    if ( !(_DWORD)j )
                      LOBYTE(j) = sub_77FA10(49);
                  }
                }
              }
            }
          }
        }
        return (char)j;
      case 10:
        if ( !v7 )
        {
          if ( v8 )
          {
            if ( !*v8 )
            {
              LODWORD(j) = sub_8D2630(v8[1], a2);
              if ( (_DWORD)j )
              {
                LODWORD(j) = sub_8D29A0(*(_QWORD *)(m + 160));
                if ( (_DWORD)j )
                {
                  if ( !strcmp(v5, "is_type") )
                  {
                    LOBYTE(j) = sub_77FA10(18);
                  }
                  else if ( !strcmp(v5, "is_alias") )
                  {
                    LOBYTE(j) = sub_77FA10(19);
                  }
                  else if ( !strcmp(v5, "is_incomplete_type") )
                  {
                    LOBYTE(j) = sub_77FA10(20);
                  }
                  else if ( !strcmp(v5, "is_template") )
                  {
                    LOBYTE(j) = sub_77FA10(21);
                  }
                  else if ( !strcmp(v5, "is_function_template") )
                  {
                    LOBYTE(j) = sub_77FA10(22);
                  }
                  else if ( !strcmp(v5, "is_variable_template") )
                  {
                    LOBYTE(j) = sub_77FA10(23);
                  }
                  else if ( !strcmp(v5, "is_class_template") )
                  {
                    LOBYTE(j) = sub_77FA10(24);
                  }
                  else if ( !strcmp(v5, "is_alias_template") )
                  {
                    LOBYTE(j) = sub_77FA10(25);
                  }
                  else if ( !strcmp(v5, "is_concept_template") )
                  {
                    LOBYTE(j) = sub_77FA10(26);
                  }
                  else if ( !strcmp(v5, "is_constant") )
                  {
                    LOBYTE(j) = sub_77FA10(27);
                  }
                  else if ( !strcmp(v5, "is_variable") )
                  {
                    LOBYTE(j) = sub_77FA10(28);
                  }
                  else if ( !strcmp(v5, "is_function") )
                  {
                    LOBYTE(j) = sub_77FA10(29);
                  }
                  else if ( !strcmp(v5, "is_namespace") )
                  {
                    LOBYTE(j) = sub_77FA10(30);
                  }
                  else if ( !strcmp(v5, "is_nsdm") )
                  {
                    LOBYTE(j) = sub_77FA10(31);
                  }
                  else if ( !strcmp(v5, "is_base") )
                  {
                    LOBYTE(j) = sub_77FA10(32);
                  }
                  else if ( !strcmp(v5, "is_constructor") )
                  {
                    LOBYTE(j) = sub_77FA10(33);
                  }
                  else if ( !strcmp(v5, "is_destructor") )
                  {
                    LOBYTE(j) = sub_77FA10(34);
                  }
                  else if ( !strcmp(v5, "is_special_member") )
                  {
                    LOBYTE(j) = sub_77FA10(35);
                  }
                  else if ( !strcmp(v5, "is_public") )
                  {
                    LOBYTE(j) = sub_77FA10(36);
                  }
                  else if ( !strcmp(v5, "is_protected") )
                  {
                    LOBYTE(j) = sub_77FA10(37);
                  }
                  else if ( !strcmp(v5, "is_private") )
                  {
                    LOBYTE(j) = sub_77FA10(38);
                  }
                  else if ( !strcmp(v5, "is_accessible") )
                  {
                    LOBYTE(j) = sub_77FA10(39);
                  }
                  else if ( !strcmp(v5, "is_static_member") )
                  {
                    LOBYTE(j) = sub_77FA10(40);
                  }
                  else if ( !strcmp(v5, "is_virtual") )
                  {
                    LOBYTE(j) = sub_77FA10(41);
                  }
                  else if ( !strcmp(v5, "is_deleted") )
                  {
                    LOBYTE(j) = sub_77FA10(42);
                  }
                  else if ( !strcmp(v5, "is_defaulted") )
                  {
                    LOBYTE(j) = sub_77FA10(43);
                  }
                  else if ( !strcmp(v5, "is_explicit") )
                  {
                    LOBYTE(j) = sub_77FA10(44);
                  }
                  else if ( !strcmp(v5, "is_override") )
                  {
                    LOBYTE(j) = sub_77FA10(45);
                  }
                  else if ( !strcmp(v5, "is_pure_virtual") )
                  {
                    LOBYTE(j) = sub_77FA10(46);
                  }
                  else
                  {
                    LODWORD(j) = strcmp(v5, "is_bit_field");
                    if ( !(_DWORD)j )
                      LOBYTE(j) = sub_77FA10(47);
                  }
                }
              }
            }
          }
        }
        return (char)j;
      case 14:
        v36 = 21;
        v37 = "make_constexpr_array";
        v38 = v5;
        do
        {
          if ( !v36 )
            break;
          v10 = *v38 < (unsigned int)*v37;
          v11 = *v38++ == *v37++;
          --v36;
        }
        while ( v11 );
        LOBYTE(j) = (!v10 && !v11) - v10;
        v39 = 0;
        v40 = (_BYTE)j == 0;
        if ( (_BYTE)j )
        {
          v47 = 14;
          v48 = "members__impl";
          v49 = (char *)v5;
          do
          {
            if ( !v47 )
              break;
            v39 = (unsigned __int8)*v49 < (unsigned __int8)*v48;
            v40 = *v49++ == *v48++;
            --v47;
          }
          while ( v40 );
          LOBYTE(j) = (!v39 && !v40) - v39;
          if ( (_BYTE)j )
          {
LABEL_244:
            if ( !v7 )
            {
              if ( v8 )
              {
                j = (_QWORD *)*v8;
                if ( *v8 )
                {
                  if ( !*j )
                  {
                    LODWORD(j) = sub_8D2630(v8[1], v49);
                    if ( (_DWORD)j )
                    {
                      LODWORD(j) = sub_641850(*(_QWORD *)(*v8 + 8LL));
                      if ( (_DWORD)j )
                      {
                        LODWORD(j) = sub_8D2630(*(_QWORD *)(m + 160), v49);
                        if ( (_DWORD)j )
                        {
                          LODWORD(j) = strcmp(v5, "metacall__impl");
                          if ( !(_DWORD)j )
                            LOBYTE(j) = sub_77FA10(62);
                        }
                      }
                    }
                  }
                }
              }
            }
            return (char)j;
          }
          if ( v7 )
            return (char)j;
        }
        else
        {
          if ( v7 )
          {
            if ( !*(_QWORD *)v7 && !*(_BYTE *)(v7 + 8) )
            {
              if ( v8 )
              {
                j = (_QWORD *)*v8;
                if ( *v8 )
                {
                  if ( !*j )
                  {
                    LODWORD(j) = sub_8D2E30(v8[1]);
                    if ( (_DWORD)j )
                    {
                      LODWORD(j) = sub_8D2780(*(_QWORD *)(*v8 + 8LL));
                      if ( (_DWORD)j )
                      {
                        LODWORD(j) = sub_8D2630(*(_QWORD *)(m + 160), v38);
                        if ( (_DWORD)j )
                          LOBYTE(j) = sub_77FA10(7);
                      }
                    }
                  }
                }
              }
            }
            return (char)j;
          }
          v49 = "members__impl";
          LODWORD(j) = strcmp(v5, "members__impl");
          if ( (_DWORD)j )
            goto LABEL_244;
        }
        if ( v8 )
        {
          if ( !*v8 )
          {
            LODWORD(j) = sub_8D2630(v8[1], v49);
            if ( (_DWORD)j )
            {
              LODWORD(j) = sub_641850(*(_QWORD *)(m + 160));
              if ( (_DWORD)j )
                LOBYTE(j) = sub_77FA10(9);
            }
          }
        }
        break;
      case 15:
        v30 = 8;
        v31 = "name_of";
        v32 = v5;
        do
        {
          if ( !v30 )
            break;
          v10 = *v32 < (unsigned int)*v31;
          v11 = *v32++ == *v31++;
          --v30;
        }
        while ( v11 );
        v33 = (!v10 && !v11) - v10;
        LOBYTE(j) = v8 != 0 && v7 == 0;
        v34 = 0;
        v35 = v33 == 0;
        if ( v33 )
        {
          v44 = 29;
          v45 = "nonstatic_data_members__impl";
          v46 = v5;
          do
          {
            if ( !v44 )
              break;
            v34 = *v46 < (unsigned int)*v45;
            v35 = *v46++ == *v45++;
            --v44;
          }
          while ( v35 );
          if ( (!v34 && !v35) == v34 )
          {
            if ( (_BYTE)j )
            {
              if ( !*v8 )
              {
                LODWORD(j) = sub_8D2630(v8[1], v46);
                if ( (_DWORD)j )
                {
                  LODWORD(j) = sub_641850(*(_QWORD *)(m + 160));
                  if ( (_DWORD)j )
                    LOBYTE(j) = sub_77FA10(11);
                }
              }
            }
          }
        }
        else if ( (_BYTE)j )
        {
          if ( !*v8 )
          {
            LODWORD(j) = sub_8D2630(v8[1], v32);
            if ( (_DWORD)j )
            {
              LODWORD(j) = sub_72CE00(*(_QWORD *)(m + 160));
              if ( (_DWORD)j )
                LOBYTE(j) = sub_77FA10(8);
            }
          }
        }
        return (char)j;
      case 16:
        v27 = 10;
        v28 = "offset_of";
        v29 = v5;
        do
        {
          if ( !v27 )
            break;
          v10 = *v29 < (unsigned int)*v28;
          v11 = *v29++ == *v28++;
          --v27;
        }
        while ( v11 );
        LOBYTE(j) = (!v10 && !v11) - v10;
        if ( v7 == 0 && v8 != 0 && (!v10 && !v11) == v10 && !*v8 )
        {
          LODWORD(j) = sub_8D2630(v8[1], v29);
          if ( (_DWORD)j )
          {
            LODWORD(j) = sub_8D3A00(*(_QWORD *)(m + 160));
            if ( (_DWORD)j )
              LOBYTE(j) = sub_77FA10(57);
          }
        }
        return (char)j;
      case 17:
        if ( !v7 )
        {
          if ( v8 )
          {
            if ( !*v8 )
            {
              LODWORD(j) = sub_8D2630(v8[1], a2);
              if ( (_DWORD)j )
              {
                LODWORD(j) = sub_8D2630(*(_QWORD *)(m + 160), a2);
                if ( (_DWORD)j )
                {
                  LODWORD(j) = strcmp(v5, "parent_of");
                  if ( !(_DWORD)j )
                    LOBYTE(j) = sub_77FA10(54);
                }
              }
            }
          }
        }
        return (char)j;
      case 19:
        v24 = 14;
        v25 = "reflect_value";
        v26 = v5;
        do
        {
          if ( !v24 )
            break;
          v10 = *v26 < (unsigned int)*v25;
          v11 = *v26++ == *v25++;
          --v24;
        }
        while ( v11 );
        LOBYTE(j) = (!v10 && !v11) - v10;
        if ( v7 )
        {
          if ( (!v10 && !v11) == v10 && !*(_QWORD *)v7 && !*(_BYTE *)(v7 + 8) )
          {
            if ( v8 )
            {
              if ( !*v8 )
              {
                LODWORD(j) = sub_8D2630(*(_QWORD *)(m + 160), v26);
                if ( (_DWORD)j )
                  LOBYTE(j) = sub_77FA10(16);
              }
            }
          }
        }
        return (char)j;
      case 20:
        v21 = 17;
        v22 = "substitute__impl";
        v23 = v5;
        do
        {
          if ( !v21 )
            break;
          v10 = *v23 < (unsigned int)*v22;
          v11 = *v23++ == *v22++;
          --v21;
        }
        while ( v11 );
        LOBYTE(j) = (!v10 && !v11) - v10;
        if ( (_BYTE)j || v7 )
        {
          if ( v8 )
          {
            if ( !(*v8 | v7) )
            {
              LODWORD(j) = sub_8D2630(v8[1], v23);
              if ( (_DWORD)j )
              {
                if ( (unsigned int)sub_8D3A00(*(_QWORD *)(m + 160)) )
                {
                  LODWORD(j) = strcmp(v5, "size_of");
                  if ( !(_DWORD)j )
                    LOBYTE(j) = sub_77FA10(56);
                }
                else
                {
                  LODWORD(j) = sub_641850(*(_QWORD *)(m + 160));
                  if ( (_DWORD)j )
                  {
                    if ( !strcmp(v5, "static_data_members__impl") )
                    {
                      LOBYTE(j) = sub_77FA10(10);
                    }
                    else
                    {
                      LODWORD(j) = strcmp(v5, "subobjects__impl");
                      if ( !(_DWORD)j )
                        LOBYTE(j) = sub_77FA10(13);
                    }
                  }
                }
              }
            }
          }
        }
        else if ( v8 )
        {
          j = (_QWORD *)*v8;
          if ( *v8 )
          {
            if ( !*j )
            {
              LODWORD(j) = sub_8D2630(v8[1], v23);
              if ( (_DWORD)j )
              {
                LODWORD(j) = sub_641850(*(_QWORD *)(*v8 + 8LL));
                if ( (_DWORD)j )
                {
                  LODWORD(j) = sub_8D2630(*(_QWORD *)(m + 160), v23);
                  if ( (_DWORD)j )
                    LOBYTE(j) = sub_77FA10(15);
                }
              }
            }
          }
        }
        return (char)j;
      case 21:
        if ( !v7 )
        {
          if ( v8 )
          {
            if ( !*v8 )
            {
              LODWORD(j) = sub_8D2630(v8[1], a2);
              if ( (_DWORD)j )
              {
                if ( (unsigned int)sub_641850(*(_QWORD *)(m + 160)) )
                {
                  LODWORD(j) = strcmp(v5, "template_arguments__impl");
                  if ( !(_DWORD)j )
                    LOBYTE(j) = sub_77FA10(51);
                }
                else
                {
                  LODWORD(j) = sub_8D2630(v20, a2);
                  if ( (_DWORD)j )
                  {
                    if ( !strcmp(v5, "template_of") )
                    {
                      LOBYTE(j) = sub_77FA10(52);
                    }
                    else
                    {
                      LODWORD(j) = strcmp(v5, "type_of");
                      if ( !(_DWORD)j )
                        LOBYTE(j) = sub_77FA10(53);
                    }
                  }
                }
              }
            }
          }
        }
        return (char)j;
      case 23:
        v17 = 9;
        v18 = "value_of";
        v19 = v5;
        do
        {
          if ( !v17 )
            break;
          v10 = *v19 < (unsigned int)*v18;
          v11 = *v19++ == *v18++;
          --v17;
        }
        while ( v11 );
        LOBYTE(j) = (!v10 && !v11) - v10;
        if ( v7 )
        {
          if ( (!v10 && !v11) == v10 && !*(_QWORD *)v7 && !*(_BYTE *)(v7 + 8) )
          {
            if ( v8 )
            {
              if ( !*v8 )
              {
                LODWORD(j) = sub_8D2630(v8[1], v19);
                if ( (_DWORD)j )
                  LOBYTE(j) = sub_77FA10(17);
              }
            }
          }
        }
        return (char)j;
      default:
        return (char)j;
    }
  }
  return (char)j;
}

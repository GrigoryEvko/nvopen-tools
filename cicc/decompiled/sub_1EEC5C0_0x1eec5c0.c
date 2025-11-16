// Function: sub_1EEC5C0
// Address: 0x1eec5c0
//
__int64 __fastcall sub_1EEC5C0(__int64 a1, _QWORD *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // r13
  __int64 v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // r14
  __int16 v10; // ax
  bool v11; // r15
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 i; // rdx
  const char *v19; // rsi
  size_t v20; // rdx
  __int64 v21; // [rsp+8h] [rbp-48h]
  unsigned __int8 v22; // [rsp+17h] [rbp-39h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_37:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4FCFA48 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_37;
  }
  v21 = *(_QWORD *)(*a2 + 40LL);
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4FCFA48);
  v7 = a2[7];
  if ( *(_BYTE *)(v7 + 65) || (v22 = *(_BYTE *)(v7 + 657)) != 0 )
  {
    v8 = (_QWORD *)a2[41];
    v22 = 0;
    while ( a2 + 40 != v8 )
    {
      v9 = v8[4];
      if ( v8 + 3 == (_QWORD *)v9 )
        goto LABEL_27;
      do
      {
        v10 = *(_WORD *)(v9 + 46);
        if ( (v10 & 4) != 0 || (v10 & 8) == 0 )
          v11 = (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 8LL) & 0x10LL) != 0;
        else
          v11 = sub_1E15D00(v9, 0x10u, 1);
        if ( !v11 )
          goto LABEL_25;
        v12 = *(_QWORD *)(v9 + 32);
        v13 = v12 + 40LL * *(unsigned int *)(v9 + 40);
        if ( v12 == v13 )
          goto LABEL_25;
        while ( 1 )
        {
          if ( *(_BYTE *)v12 == 10 )
          {
            if ( *(_BYTE *)(*(_QWORD *)(v12 + 24) + 16LL) )
              goto LABEL_25;
LABEL_19:
            v14 = sub_210D800(v6);
            if ( v15 )
            {
              v16 = *(_QWORD *)(v9 + 32);
              for ( i = v16 + 40LL * *(unsigned int *)(v9 + 40); i != v16; v16 += 40 )
              {
                if ( *(_BYTE *)v16 == 12 )
                  *(_QWORD *)(v16 + 24) = v14;
              }
              v22 = v11;
            }
            goto LABEL_25;
          }
          if ( *(_BYTE *)v12 == 9 )
            break;
          v12 += 40;
          if ( v13 == v12 )
            goto LABEL_25;
        }
        v19 = *(const char **)(v12 + 24);
        v20 = 0;
        if ( v19 )
        {
          v19 = *(const char **)(v12 + 24);
          v20 = strlen(v19);
        }
        if ( sub_16321A0(v21, (__int64)v19, v20) )
          goto LABEL_19;
LABEL_25:
        if ( (*(_BYTE *)v9 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v9 + 46) & 8) != 0 )
            v9 = *(_QWORD *)(v9 + 8);
        }
        v9 = *(_QWORD *)(v9 + 8);
      }
      while ( v8 + 3 != (_QWORD *)v9 );
LABEL_27:
      v8 = (_QWORD *)v8[1];
    }
  }
  return v22;
}

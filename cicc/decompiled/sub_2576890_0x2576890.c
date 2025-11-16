// Function: sub_2576890
// Address: 0x2576890
//
_QWORD *__fastcall sub_2576890(__int64 a1, int a2)
{
  __int64 *v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  __int64 *v6; // r12
  _QWORD *v7; // rax
  _QWORD *i; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdx
  _QWORD *result; // rax
  __int64 v13; // r10
  int v14; // edi
  unsigned __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 *v17; // rsi
  __int64 v18; // r9
  char v19; // al
  __int64 *v20; // rax
  _QWORD *j; // rcx
  __int64 *v22; // [rsp+8h] [rbp-98h]
  __int64 *v23; // [rsp+10h] [rbp-90h]
  __int64 v24; // [rsp+18h] [rbp-88h]
  __int64 v25; // [rsp+18h] [rbp-88h]
  __int64 v26; // [rsp+28h] [rbp-78h]
  __int64 *v27; // [rsp+30h] [rbp-70h] BYREF
  __int64 v28; // [rsp+38h] [rbp-68h]
  __int64 *v29; // [rsp+40h] [rbp-60h]
  __int64 v30; // [rsp+48h] [rbp-58h]
  __int64 *v31[10]; // [rsp+50h] [rbp-50h] BYREF

  v3 = *(__int64 **)(a1 + 8);
  v4 = *(unsigned int *)(a1 + 24);
  v26 = (__int64)v3;
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  *(_QWORD *)(a1 + 8) = sub_C7D670(8LL * v5, 8);
  if ( v3 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v6 = &v3[v4];
    if ( !byte_4FEF240[0] && (unsigned int)sub_2207590((__int64)byte_4FEF240) )
    {
      qword_4FEF260 = -4096;
      unk_4FEF268 = -4096;
      qword_4FEF270 = 0;
      unk_4FEF278 = 0;
      sub_2207640((__int64)byte_4FEF240);
    }
    v7 = *(_QWORD **)(a1 + 8);
    for ( i = &v7[*(unsigned int *)(a1 + 24)]; i != v7; ++v7 )
    {
      if ( v7 )
        *v7 = &qword_4FEF260;
    }
    if ( !byte_4FEF240[0] && (unsigned int)sub_2207590((__int64)byte_4FEF240) )
    {
      qword_4FEF260 = -4096;
      unk_4FEF268 = -4096;
      qword_4FEF270 = 0;
      unk_4FEF278 = 0;
      sub_2207640((__int64)byte_4FEF240);
    }
    if ( !byte_4FEF208[0] && (unsigned int)sub_2207590((__int64)byte_4FEF208) )
    {
      qword_4FEF220 = -8192;
      unk_4FEF228 = -8192;
      qword_4FEF230 = 0;
      unk_4FEF238 = 0;
      sub_2207640((__int64)byte_4FEF208);
    }
    while ( v6 != v3 )
    {
      v9 = (_QWORD *)*v3;
      v10 = *(_QWORD *)*v3;
      v11 = *(_QWORD *)(*v3 + 8);
      if ( unk_4FEF268 != v11 || qword_4FEF260 != v10 )
        goto LABEL_14;
      v13 = v9[2];
      if ( v13 != qword_4FEF270 )
      {
        if ( v13 == -4096 || qword_4FEF270 == -4096 || qword_4FEF270 == -8192 || v13 == -8192 )
          goto LABEL_14;
        if ( v13 )
        {
          v14 = *(_DWORD *)(v13 + 20) - *(_DWORD *)(v13 + 24);
          if ( qword_4FEF270 )
          {
            v24 = qword_4FEF270;
            if ( v14 != *(_DWORD *)(qword_4FEF270 + 20) - *(_DWORD *)(qword_4FEF270 + 24) )
              goto LABEL_14;
            if ( v14 )
            {
              v23 = (__int64 *)v9[2];
              v22 = *(__int64 **)(v13 + 8);
              v28 = sub_254BB00(v13);
              v27 = v22;
              sub_254BBF0((__int64)&v27);
              v29 = v23;
              v30 = *v23;
              v31[0] = (__int64 *)sub_254BB00((__int64)v23);
              v31[1] = v31[0];
              sub_254BBF0((__int64)v31);
              v17 = v27;
              v18 = v24;
              v31[2] = v23;
              v31[3] = (__int64 *)*v23;
              if ( v31[0] != v27 )
              {
                while ( 1 )
                {
                  v25 = v18;
                  v19 = sub_B19060(v18, *v17, v15, v16);
                  v18 = v25;
                  if ( !v19 )
                    break;
                  v17 = (__int64 *)v28;
                  v20 = v27 + 1;
                  v27 = v20;
                  if ( v20 != (__int64 *)v28 )
                  {
                    while ( 1 )
                    {
                      v16 = *v20;
                      v15 = *v20 + 2;
                      if ( v15 > 1 )
                        break;
                      v27 = ++v20;
                      if ( v20 == (__int64 *)v28 )
                        goto LABEL_31;
                    }
                    v17 = v27;
                  }
LABEL_31:
                  if ( v31[0] == v17 )
                    goto LABEL_17;
                }
                v9 = (_QWORD *)*v3;
                v10 = *(_QWORD *)*v3;
                v11 = *(_QWORD *)(*v3 + 8);
LABEL_14:
                if ( qword_4FEF220 != v10 || unk_4FEF228 != v11 || !sub_254C7C0((__int64 *)v9[2], qword_4FEF230) )
                {
                  sub_25682F0(a1, v3, v31);
                  *v31[0] = *v3;
                  ++*(_DWORD *)(a1 + 16);
                }
              }
            }
          }
          else if ( v14 )
          {
            goto LABEL_14;
          }
        }
        else if ( qword_4FEF270 && *(_DWORD *)(qword_4FEF270 + 24) != *(_DWORD *)(qword_4FEF270 + 20) )
        {
          goto LABEL_14;
        }
      }
LABEL_17:
      ++v3;
    }
    return (_QWORD *)sub_C7D6A0(v26, 8 * v4, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    if ( !byte_4FEF240[0] && (unsigned int)sub_2207590((__int64)byte_4FEF240) )
    {
      qword_4FEF260 = -4096;
      unk_4FEF268 = -4096;
      qword_4FEF270 = 0;
      unk_4FEF278 = 0;
      sub_2207640((__int64)byte_4FEF240);
    }
    result = *(_QWORD **)(a1 + 8);
    for ( j = &result[*(unsigned int *)(a1 + 24)]; j != result; ++result )
    {
      if ( result )
        *result = &qword_4FEF260;
    }
  }
  return result;
}

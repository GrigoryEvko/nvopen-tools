// Function: sub_3026120
// Address: 0x3026120
//
void __fastcall sub_3026120(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  char *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 i; // r12
  __int64 v12; // rsi
  __int64 v13; // r14
  unsigned __int8 *v14; // r15
  unsigned __int8 v15; // al
  __int64 v16; // rsi
  char *v17; // rax
  __int64 v18; // rax
  char *v19; // rax
  __int64 v21; // [rsp+18h] [rbp-178h]
  __int64 v23; // [rsp+38h] [rbp-158h] BYREF
  __int64 v24; // [rsp+40h] [rbp-150h] BYREF
  char *v25; // [rsp+48h] [rbp-148h]
  __int64 v26; // [rsp+50h] [rbp-140h]
  int v27; // [rsp+58h] [rbp-138h]
  char v28; // [rsp+5Ch] [rbp-134h]
  char v29; // [rsp+60h] [rbp-130h] BYREF

  v25 = &v29;
  v3 = *(_QWORD *)(a2 + 32);
  v24 = 0;
  v26 = 32;
  v27 = 0;
  v28 = 1;
  v21 = a2 + 24;
  if ( v3 == a2 + 24 )
  {
    v10 = *(_QWORD *)(a2 + 48);
    i = a2 + 40;
    if ( v10 != a2 + 40 )
      goto LABEL_11;
    return;
  }
LABEL_7:
  while ( 2 )
  {
    if ( !v3 )
      BUG();
    v8 = v3 - 56;
    v23 = *(_QWORD *)(v3 + 64);
    if ( !(unsigned __int8)sub_A747A0(&v23, "nvptx-libcall-callee", 0x14u) )
    {
      if ( sub_B2FC80(v3 - 56) )
      {
        if ( !*(_QWORD *)(v3 - 40) || *(_DWORD *)(v3 - 20) )
          goto LABEL_6;
        goto LABEL_9;
      }
      v13 = *(_QWORD *)(v3 - 40);
      if ( !v13 )
        goto LABEL_30;
      while ( 1 )
      {
        v14 = *(unsigned __int8 **)(v13 + 24);
        v15 = *v14;
        if ( *v14 <= 0x15u )
        {
          if ( (unsigned __int8)sub_3020900(*(_QWORD *)(v13 + 24))
            || (unsigned __int8)sub_30200E0((__int64)v14, (__int64)&v24) )
          {
            goto LABEL_29;
          }
          v15 = *v14;
        }
        if ( v15 > 0x1Cu )
        {
          v16 = sub_B43CB0((__int64)v14);
          if ( v16 )
          {
            if ( v28 )
            {
              v17 = v25;
              v5 = (__int64)&v25[8 * HIDWORD(v26)];
              if ( v25 != (char *)v5 )
              {
                while ( v16 != *(_QWORD *)v17 )
                {
                  v17 += 8;
                  if ( (char *)v5 == v17 )
                    goto LABEL_38;
                }
LABEL_29:
                v18 = sub_31DB510(a1, v3 - 56);
                sub_3025D20(a1, v3 - 56, v18, a3);
LABEL_30:
                if ( !v28 )
                  goto LABEL_36;
                v19 = v25;
                v5 = HIDWORD(v26);
                v4 = &v25[8 * HIDWORD(v26)];
                if ( v25 == v4 )
                {
LABEL_34:
                  if ( HIDWORD(v26) < (unsigned int)v26 )
                  {
                    ++HIDWORD(v26);
                    *(_QWORD *)v4 = v8;
                    ++v24;
                    goto LABEL_6;
                  }
LABEL_36:
                  sub_C8CC70((__int64)&v24, v3 - 56, (__int64)v4, v5, v6, v7);
                  goto LABEL_6;
                }
                while ( v8 != *(_QWORD *)v19 )
                {
                  v19 += 8;
                  if ( v4 == v19 )
                    goto LABEL_34;
                }
LABEL_6:
                v3 = *(_QWORD *)(v3 + 8);
                if ( v21 == v3 )
                  goto LABEL_10;
                goto LABEL_7;
              }
            }
            else if ( sub_C8CA60((__int64)&v24, v16) )
            {
              goto LABEL_29;
            }
          }
        }
LABEL_38:
        v13 = *(_QWORD *)(v13 + 8);
        if ( !v13 )
          goto LABEL_30;
      }
    }
LABEL_9:
    v9 = sub_31DB510(a1, v3 - 56);
    sub_3025D20(a1, v3 - 56, v9, a3);
    v3 = *(_QWORD *)(v3 + 8);
    if ( v21 != v3 )
      continue;
    break;
  }
LABEL_10:
  v10 = *(_QWORD *)(a2 + 48);
  for ( i = a2 + 40; v10 != i; v10 = *(_QWORD *)(v10 + 8) )
  {
LABEL_11:
    v12 = v10 - 48;
    if ( !v10 )
      v12 = 0;
    sub_3026070(a1, v12, a3);
  }
  if ( !v28 )
    _libc_free((unsigned __int64)v25);
}

// Function: sub_2151D30
// Address: 0x2151d30
//
__int64 __fastcall sub_2151D30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  _BYTE *v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // r11
  __int64 v14; // r10
  const char *v15; // rax
  size_t v16; // rdx
  __int64 v17; // r10
  char *v18; // rsi
  _WORD *v19; // rdi
  unsigned __int64 v20; // rax
  const char *v21; // rax
  size_t v22; // rdx
  __int64 v23; // r10
  char *v24; // rsi
  _WORD *v25; // rdi
  unsigned __int64 v26; // rax
  _WORD *v27; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rax
  const char *v36; // rax
  __int64 v37; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+8h] [rbp-78h]
  __int64 v39; // [rsp+10h] [rbp-70h]
  __int64 v40; // [rsp+10h] [rbp-70h]
  size_t v41; // [rsp+10h] [rbp-70h]
  size_t v42; // [rsp+10h] [rbp-70h]
  __int64 v43; // [rsp+18h] [rbp-68h]
  char *v44; // [rsp+20h] [rbp-60h] BYREF
  size_t v45; // [rsp+28h] [rbp-58h]
  _QWORD v46[2]; // [rsp+30h] [rbp-50h] BYREF
  char v47; // [rsp+40h] [rbp-40h]

  if ( !sub_15E4F60(a2) )
  {
    if ( *(_QWORD *)(a1 + 840) )
      goto LABEL_3;
    goto LABEL_64;
  }
  v33 = sub_3936750();
  if ( (unsigned __int8)sub_39371E0(a2, v33) )
  {
    v36 = (const char *)sub_3936860(v33, 0);
    sub_214B770((__int64 *)&v44, v36);
    sub_16E7EE0(a3, v44, v45);
    if ( v44 != (char *)v46 )
      j_j___libc_free_0(v44, v46[0] + 1LL);
  }
  sub_39367A0(v33);
  if ( !*(_QWORD *)(a1 + 840) )
LABEL_64:
    *(_QWORD *)(a1 + 840) = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 232) + 16LL))(
                              *(_QWORD *)(a1 + 232),
                              a2);
LABEL_3:
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 232) + 952LL) == 1 )
    sub_214CAD0((_BYTE *)a2, a3);
  if ( (unsigned __int8)sub_1C2F070(a2) )
  {
    sub_1263B40(a3, ".entry ");
    if ( !(unsigned __int8)sub_1C2FA50(a2) )
      goto LABEL_7;
  }
  else
  {
    sub_1263B40(a3, ".func ");
    if ( !(unsigned __int8)sub_1C2FA50(a2) )
      goto LABEL_7;
  }
  sub_214C940(a2, a3);
LABEL_7:
  sub_214D1D0(a1, **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL), a2, a3);
  v6 = sub_396EAF0(a1, a2);
  sub_38E2490(v6, a3, *(_QWORD *)(a1 + 240));
  v7 = *(_BYTE **)(a3 + 24);
  if ( *(_BYTE **)(a3 + 16) == v7 )
  {
    sub_16E7EE0(a3, "\n", 1u);
  }
  else
  {
    *v7 = 10;
    ++*(_QWORD *)(a3 + 24);
  }
  sub_21502D0((_QWORD *)a1, a2, a3);
  v8 = (_QWORD *)(a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( ((a2 >> 2) & 1) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v8 + 7, -1, 29) )
      goto LABEL_65;
  }
  else
  {
    if ( v8 )
    {
      if ( !(unsigned __int8)sub_1560180((__int64)(v8 + 14), 29) )
        goto LABEL_12;
      goto LABEL_46;
    }
    if ( (unsigned __int8)sub_1560260((_QWORD *)0x38, -1, 29) )
    {
LABEL_38:
      if ( *(_BYTE *)(**(_QWORD **)(MEMORY[0x40] + 16LL) + 8LL) )
        goto LABEL_12;
LABEL_39:
      if ( (unsigned __int8)sub_1C2F070((__int64)v8) )
        goto LABEL_12;
LABEL_66:
      sub_1263B40(a3, ".noreturn ");
      goto LABEL_12;
    }
  }
  v29 = *(v8 - 3);
  if ( !*(_BYTE *)(v29 + 16) )
  {
    v44 = *(char **)(v29 + 112);
    if ( (unsigned __int8)sub_1560260(&v44, -1, 29) )
    {
      if ( ((a2 >> 2) & 1) == 0 )
      {
        if ( v8 )
        {
LABEL_46:
          if ( *(_BYTE *)(**(_QWORD **)(v8[3] + 16LL) + 8LL) )
          {
            if ( !sub_15E4F60(a2) )
              goto LABEL_13;
            goto LABEL_48;
          }
          goto LABEL_39;
        }
        goto LABEL_38;
      }
LABEL_65:
      if ( *(_BYTE *)(**(_QWORD **)(v8[8] + 16LL) + 8LL) )
        goto LABEL_12;
      goto LABEL_66;
    }
  }
LABEL_12:
  if ( !sub_15E4F60(a2) )
    goto LABEL_13;
LABEL_48:
  sub_3937CA0(&v44, a2, 0);
  if ( v47 )
  {
    sub_16E7EE0(a3, v44, v45);
    if ( v47 )
    {
      if ( v44 != (char *)v46 )
        j_j___libc_free_0(v44, v46[0] + 1LL);
    }
  }
LABEL_13:
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_QWORD *)(v9 + 48);
  v43 = v9 + 40;
  if ( v9 + 40 != v10 )
  {
    while ( 1 )
    {
      if ( !v10 )
        BUG();
      v11 = *(_QWORD *)(v10 - 72);
      if ( !v11 )
        BUG();
      if ( *(_BYTE *)(v11 + 16) == 5 && *(_WORD *)(v11 + 18) == 47 )
      {
        v11 = *(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
        if ( !v11 )
          goto LABEL_15;
      }
      if ( v11 != a2 )
        goto LABEL_15;
      sub_2151550((_QWORD *)a1, v10 - 48, v11, a3);
      v12 = *(_QWORD *)(a3 + 24);
      v13 = v10 - 48;
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v12) <= 6 )
      {
        v32 = sub_16E7EE0(a3, ".alias ", 7u);
        v13 = v10 - 48;
        v14 = v32;
      }
      else
      {
        *(_DWORD *)v12 = 1768710446;
        v14 = a3;
        *(_WORD *)(v12 + 4) = 29537;
        *(_BYTE *)(v12 + 6) = 32;
        *(_QWORD *)(a3 + 24) += 7LL;
      }
      v39 = v14;
      v15 = sub_1649960(v13);
      v17 = v39;
      v18 = (char *)v15;
      v19 = *(_WORD **)(v39 + 24);
      v20 = *(_QWORD *)(v39 + 16) - (_QWORD)v19;
      if ( v16 > v20 )
      {
        v31 = sub_16E7EE0(v39, v18, v16);
        v19 = *(_WORD **)(v31 + 24);
        v17 = v31;
        v20 = *(_QWORD *)(v31 + 16) - (_QWORD)v19;
      }
      else if ( v16 )
      {
        v38 = v39;
        v42 = v16;
        memcpy(v19, v18, v16);
        v17 = v38;
        v35 = *(_QWORD *)(v38 + 16);
        v19 = (_WORD *)(*(_QWORD *)(v38 + 24) + v42);
        *(_QWORD *)(v38 + 24) = v19;
        v20 = v35 - (_QWORD)v19;
      }
      if ( v20 <= 1 )
      {
        v17 = sub_16E7EE0(v17, ", ", 2u);
      }
      else
      {
        *v19 = 8236;
        *(_QWORD *)(v17 + 24) += 2LL;
      }
      v40 = v17;
      v21 = sub_1649960(v11);
      v23 = v40;
      v24 = (char *)v21;
      v25 = *(_WORD **)(v40 + 24);
      v26 = *(_QWORD *)(v40 + 16) - (_QWORD)v25;
      if ( v22 > v26 )
      {
        v30 = sub_16E7EE0(v40, v24, v22);
        v25 = *(_WORD **)(v30 + 24);
        v23 = v30;
        v26 = *(_QWORD *)(v30 + 16) - (_QWORD)v25;
      }
      else if ( v22 )
      {
        v37 = v40;
        v41 = v22;
        memcpy(v25, v24, v22);
        v23 = v37;
        v34 = *(_QWORD *)(v37 + 16);
        v25 = (_WORD *)(*(_QWORD *)(v37 + 24) + v41);
        *(_QWORD *)(v37 + 24) = v25;
        v26 = v34 - (_QWORD)v25;
      }
      if ( v26 <= 1 )
      {
        sub_16E7EE0(v23, ";\n", 2u);
LABEL_15:
        v10 = *(_QWORD *)(v10 + 8);
        if ( v43 == v10 )
          break;
      }
      else
      {
        *v25 = 2619;
        *(_QWORD *)(v23 + 24) += 2LL;
        v10 = *(_QWORD *)(v10 + 8);
        if ( v43 == v10 )
          break;
      }
    }
  }
  v27 = *(_WORD **)(a3 + 24);
  if ( *(_QWORD *)(a3 + 16) - (_QWORD)v27 <= 1u )
    return sub_16E7EE0(a3, ";\n", 2u);
  *v27 = 2619;
  *(_QWORD *)(a3 + 24) += 2LL;
  return 2619;
}

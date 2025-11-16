// Function: sub_3914E60
// Address: 0x3914e60
//
char __fastcall sub_3914E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, char a6)
{
  __int64 v9; // r12
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  void *v13; // r13
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rbx
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  char v22; // al
  __int64 v23; // rdi
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rsi
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+18h] [rbp-38h]

  if ( a5 )
    return 1;
  v9 = sub_3914E30(a1, a3);
  v10 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10
    || (*(_BYTE *)(v9 + 9) & 0xC) == 8
    && (*(_BYTE *)(v9 + 8) |= 4u,
        v10 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24)),
        *(_QWORD *)v9 = v10 | *(_QWORD *)v9 & 7LL,
        v10) )
  {
    v11 = *(_QWORD *)(v10 + 24);
  }
  else
  {
    v11 = 0;
  }
  v12 = *(_QWORD *)(a4 + 24);
  if ( !a6 )
    goto LABEL_9;
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 12LL) == (_DWORD)&algn_1000005[2] )
  {
    v13 = *(void **)(a4 + 32);
    if ( !v13 )
    {
      v22 = *(_BYTE *)(v9 + 8);
      if ( (v22 & 1) != 0 )
      {
        if ( (*(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v13 = (void *)(*(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL);
        }
        else
        {
          if ( (*(_BYTE *)(v9 + 9) & 0xC) != 8 )
            return 0;
          v23 = *(_QWORD *)(v9 + 24);
          v31 = v11;
          *(_BYTE *)(v9 + 8) = v22 | 4;
          v33 = v12;
          v24 = (unsigned __int64)sub_38CE440(v23);
          v12 = v33;
          v11 = v31;
          v25 = v24;
          v26 = v24 | *(_QWORD *)v9 & 7LL;
          *(_QWORD *)v9 = v26;
          if ( !v25 )
            goto LABEL_9;
          v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v27 )
          {
            v13 = (void *)v27;
          }
          else if ( (*(_BYTE *)(v9 + 9) & 0xC) == 8 )
          {
            *(_BYTE *)(v9 + 8) |= 4u;
            v28 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24));
            v11 = v31;
            v12 = v33;
            v13 = (void *)v28;
            *(_QWORD *)v9 = v28 | *(_QWORD *)v9 & 7LL;
          }
        }
        if ( off_4CF6DB8 != v13 )
          return v12 == v11;
      }
    }
LABEL_9:
    if ( v12 == v11 )
    {
      v21 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v21 )
        return *(_QWORD *)(v21 + 32) == *(_QWORD *)(a4 + 32);
      if ( (*(_BYTE *)(v9 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v9 + 8) |= 4u;
        v21 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24));
        *(_QWORD *)v9 = v21 | *(_QWORD *)v9 & 7LL;
        if ( v21 )
          return *(_QWORD *)(v21 + 32) == *(_QWORD *)(a4 + 32);
      }
    }
    return 0;
  }
  v14 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v14 )
  {
    if ( (*(_BYTE *)(v9 + 9) & 0xC) != 8 )
      return 0;
    *(_BYTE *)(v9 + 8) |= 4u;
    v29 = v11;
    v30 = v12;
    v15 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24));
    v16 = v15 | *(_QWORD *)v9 & 7LL;
    *(_QWORD *)v9 = v16;
    if ( !v15 )
      return 0;
    v14 = v16 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = v30;
    v11 = v29;
    if ( !v14 )
    {
      v14 = 0;
      if ( (*(_BYTE *)(v9 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v9 + 8) |= 4u;
        v14 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v9 + 24));
        v11 = v29;
        v12 = v30;
        *(_QWORD *)v9 = v14 | *(_QWORD *)v9 & 7LL;
      }
    }
  }
  if ( off_4CF6DB8 == (_UNKNOWN *)v14 || v12 != v11 )
    return 0;
  v17 = *(_BYTE *)(v9 + 8);
  if ( (v17 & 1) != 0 )
    return 1;
  v18 = *(_QWORD *)(a4 + 32);
  v19 = *(_QWORD *)v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v19 )
  {
    if ( (*(_BYTE *)(v9 + 9) & 0xC) != 8 )
      BUG();
    v20 = *(_QWORD *)(v9 + 24);
    *(_BYTE *)(v9 + 8) = v17 | 4;
    v19 = (unsigned __int64)sub_38CE440(v20);
    *(_QWORD *)v9 = v19 | *(_QWORD *)v9 & 7LL;
  }
  if ( v18 == *(_QWORD *)(v19 + 32) )
    return 1;
  return (*(_BYTE *)(a2 + 484) & 2) == 0;
}

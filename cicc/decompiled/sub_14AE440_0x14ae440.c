// Function: sub_14AE440
// Address: 0x14ae440
//
__int64 __fastcall sub_14AE440(__int64 a1)
{
  unsigned int v1; // r15d
  unsigned __int8 v3; // al
  __int64 v5; // r12
  unsigned __int64 v6; // r13
  unsigned __int64 v7; // rdi
  int v8; // r12d
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-48h]
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+8h] [rbp-48h]
  __int64 v48; // [rsp+8h] [rbp-48h]
  __int64 v49; // [rsp+8h] [rbp-48h]
  __int64 v50; // [rsp+8h] [rbp-48h]
  _QWORD v51[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_BYTE *)(a1 + 16);
  if ( (v3 & 0xFD) == 0x20 )
    return *(_WORD *)(a1 + 18) & 1;
  LOBYTE(v1) = v3 == 25 || (unsigned __int8)(v3 - 30) <= 1u;
  if ( (_BYTE)v1 )
    return 0;
  if ( v3 <= 0x17u )
    return 1;
  v5 = a1 | 4;
  if ( v3 != 78 )
  {
    if ( v3 != 29 )
      return 1;
    v5 = a1 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v5 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 1;
  v7 = v6 + 56;
  v8 = (v5 >> 2) & 1;
  if ( v8 )
  {
    if ( (unsigned __int8)sub_1560260(v7, 0xFFFFFFFFLL, 30)
      || (v9 = *(_QWORD *)(v6 - 24), !*(_BYTE *)(v9 + 16))
      && (v51[0] = *(_QWORD *)(v9 + 112), (unsigned __int8)sub_1560260(v51, 0xFFFFFFFFLL, 30)) )
    {
      if ( (unsigned __int8)sub_1560260(v6 + 56, 0xFFFFFFFFLL, 36) )
        return 1;
      if ( *(char *)(v6 + 23) >= 0 )
        goto LABEL_86;
      v10 = sub_1648A40(v6);
      v12 = v10 + v11;
      v13 = 0;
      if ( *(char *)(v6 + 23) < 0 )
      {
        v45 = v12;
        v14 = sub_1648A40(v6);
        v12 = v45;
        v13 = v14;
      }
      if ( !(unsigned int)((v12 - v13) >> 4) )
      {
LABEL_86:
        v15 = *(_QWORD *)(v6 - 24);
        if ( !*(_BYTE *)(v15 + 16) )
        {
          v51[0] = *(_QWORD *)(v15 + 112);
          if ( (unsigned __int8)sub_1560260(v51, 0xFFFFFFFFLL, 36) )
            return 1;
        }
      }
      if ( (unsigned __int8)sub_1560260(v6 + 56, 0xFFFFFFFFLL, 37) )
        return 1;
      if ( *(char *)(v6 + 23) >= 0
        || ((v16 = sub_1648A40(v6), v18 = v16 + v17, *(char *)(v6 + 23) >= 0)
          ? (v19 = 0)
          : (v46 = v16 + v17, v19 = sub_1648A40(v6), v18 = v46),
            v19 == v18) )
      {
LABEL_44:
        v28 = *(_QWORD *)(v6 - 24);
        if ( !*(_BYTE *)(v28 + 16) )
        {
          v51[0] = *(_QWORD *)(v28 + 112);
          if ( (unsigned __int8)sub_1560260(v51, 0xFFFFFFFFLL, 37) )
            return 1;
        }
      }
      else
      {
        while ( *(_DWORD *)(*(_QWORD *)v19 + 8LL) <= 1u )
        {
          v19 += 16;
          if ( v18 == v19 )
            goto LABEL_44;
        }
      }
      if ( !(unsigned __int8)sub_1560260(v6 + 56, 0xFFFFFFFFLL, 4) )
      {
        if ( *(char *)(v6 + 23) < 0 )
        {
          v20 = sub_1648A40(v6);
          v22 = v21 + v20;
          v23 = 0;
          v47 = v22;
          if ( *(char *)(v6 + 23) < 0 )
            v23 = sub_1648A40(v6);
          if ( (unsigned int)((v47 - v23) >> 4) )
            goto LABEL_85;
        }
        v24 = *(_QWORD *)(v6 - 24);
        if ( *(_BYTE *)(v24 + 16) )
        {
LABEL_85:
          if ( *(_BYTE *)(a1 + 16) != 78 || (v25 = *(_QWORD *)(a1 - 24), *(_BYTE *)(v25 + 16)) )
          {
LABEL_40:
            if ( !(unsigned __int8)sub_1560260(v6 + 56, 0xFFFFFFFFLL, 8) )
            {
              v27 = *(_QWORD *)(v6 - 24);
              if ( *(_BYTE *)(v27 + 16) )
                return v1;
LABEL_42:
              v51[0] = *(_QWORD *)(v27 + 112);
              return (unsigned int)sub_1560260(v51, 0xFFFFFFFFLL, 8);
            }
            return 1;
          }
          goto LABEL_37;
        }
LABEL_34:
        v51[0] = *(_QWORD *)(v24 + 112);
        if ( (unsigned __int8)sub_1560260(v51, 0xFFFFFFFFLL, 4) )
          return 1;
        if ( *(_BYTE *)(a1 + 16) != 78 || (v25 = *(_QWORD *)(a1 - 24), *(_BYTE *)(v25 + 16)) )
        {
LABEL_39:
          if ( (_BYTE)v8 )
            goto LABEL_40;
          goto LABEL_71;
        }
LABEL_37:
        v26 = *(_DWORD *)(v25 + 36);
        if ( v26 == 4 || v26 == 191 )
          return 1;
        goto LABEL_39;
      }
      return 1;
    }
  }
  else if ( (unsigned __int8)sub_1560260(v7, 0xFFFFFFFFLL, 30)
         || (v29 = *(_QWORD *)(v6 - 72), !*(_BYTE *)(v29 + 16))
         && (v51[0] = *(_QWORD *)(v29 + 112), (unsigned __int8)sub_1560260(v51, 0xFFFFFFFFLL, 30)) )
  {
    if ( (unsigned __int8)sub_1560260(v6 + 56, 0xFFFFFFFFLL, 36) )
      return 1;
    if ( *(char *)(v6 + 23) >= 0 )
      goto LABEL_88;
    v30 = sub_1648A40(v6);
    v32 = v30 + v31;
    v33 = 0;
    if ( *(char *)(v6 + 23) < 0 )
    {
      v48 = v32;
      v34 = sub_1648A40(v6);
      v32 = v48;
      v33 = v34;
    }
    if ( !(unsigned int)((v32 - v33) >> 4) )
    {
LABEL_88:
      v35 = *(_QWORD *)(v6 - 72);
      if ( !*(_BYTE *)(v35 + 16) )
      {
        v51[0] = *(_QWORD *)(v35 + 112);
        if ( (unsigned __int8)sub_1560260(v51, 0xFFFFFFFFLL, 36) )
          return 1;
      }
    }
    if ( (unsigned __int8)sub_1560260(v6 + 56, 0xFFFFFFFFLL, 37) )
      return 1;
    if ( *(char *)(v6 + 23) >= 0
      || ((v36 = sub_1648A40(v6), v38 = v36 + v37, *(char *)(v6 + 23) >= 0)
        ? (v39 = 0)
        : (v49 = v36 + v37, v39 = sub_1648A40(v6), v38 = v49),
          v39 == v38) )
    {
LABEL_75:
      v44 = *(_QWORD *)(v6 - 72);
      if ( !*(_BYTE *)(v44 + 16) )
      {
        v51[0] = *(_QWORD *)(v44 + 112);
        if ( (unsigned __int8)sub_1560260(v51, 0xFFFFFFFFLL, 37) )
          return 1;
      }
    }
    else
    {
      while ( *(_DWORD *)(*(_QWORD *)v39 + 8LL) <= 1u )
      {
        v39 += 16;
        if ( v38 == v39 )
          goto LABEL_75;
      }
    }
    if ( !(unsigned __int8)sub_1560260(v6 + 56, 0xFFFFFFFFLL, 4) )
    {
      if ( *(char *)(v6 + 23) < 0 )
      {
        v40 = sub_1648A40(v6);
        v42 = v41 + v40;
        v43 = 0;
        v50 = v42;
        if ( *(char *)(v6 + 23) < 0 )
          v43 = sub_1648A40(v6);
        if ( (unsigned int)((v50 - v43) >> 4) )
          goto LABEL_87;
      }
      v24 = *(_QWORD *)(v6 - 72);
      if ( *(_BYTE *)(v24 + 16) )
      {
LABEL_87:
        if ( *(_BYTE *)(a1 + 16) != 78 || (v25 = *(_QWORD *)(a1 - 24), *(_BYTE *)(v25 + 16)) )
        {
LABEL_71:
          if ( !(unsigned __int8)sub_1560260(v6 + 56, 0xFFFFFFFFLL, 8) )
          {
            v27 = *(_QWORD *)(v6 - 72);
            if ( *(_BYTE *)(v27 + 16) )
              return v1;
            goto LABEL_42;
          }
          return 1;
        }
        goto LABEL_37;
      }
      goto LABEL_34;
    }
    return 1;
  }
  return v1;
}

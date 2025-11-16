// Function: sub_134FA60
// Address: 0x134fa60
//
__int64 __fastcall sub_134FA60(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  unsigned __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r15
  int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // r15d
  __int64 v15; // rax
  unsigned int v16; // r13d
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  int v20; // r14d
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // r14d
  __int64 v24; // rax
  __int64 v25; // rdx
  _DWORD *v26; // rax
  unsigned int v27; // edx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  _DWORD *v32; // r14
  _DWORD *v33; // rax
  unsigned int v34; // edx
  _QWORD v35[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a3;
  v4 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = (*a1 >> 2) & 1;
  if ( ((*a1 >> 2) & 1) == 0 )
  {
    if ( !(_DWORD)a2 )
    {
      if ( !(unsigned __int8)sub_1560260(v4 + 56, a2, a3) )
      {
        v6 = *(_QWORD *)(v4 - 72);
        if ( !*(_BYTE *)(v6 + 16) )
          goto LABEL_5;
        goto LABEL_9;
      }
LABEL_36:
      LODWORD(v5) = 1;
      return (unsigned int)v5;
    }
    v7 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
    if ( *(char *)(v4 + 23) < 0 )
    {
      v8 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
      v10 = v8 + v9;
      if ( *(char *)(v4 + 23) >= 0 )
      {
        if ( (unsigned int)(v10 >> 4) )
          goto LABEL_55;
      }
      else if ( (unsigned int)((v10 - sub_1648A40(v4)) >> 4) )
      {
        if ( *(char *)(v4 + 23) >= 0 )
          goto LABEL_55;
        v11 = *(_DWORD *)(sub_1648A40(v4) + 8);
        if ( *(char *)(v4 + 23) >= 0 )
          goto LABEL_58;
        v12 = sub_1648A40(v4);
        LODWORD(v12) = *(_DWORD *)(v12 + v13 - 4) - v11;
        v14 = a2 - 1;
        if ( (unsigned int)a2 < v7 - 2 - (int)v12 )
        {
LABEL_16:
          if ( !(unsigned __int8)sub_1560290(v4 + 56, v14, v3) )
          {
            v15 = *(_QWORD *)(v4 - 72);
            if ( !*(_BYTE *)(v15 + 16) )
            {
              v35[0] = *(_QWORD *)(v15 + 112);
              LODWORD(v5) = sub_1560290(v35, v14, v3);
              return (unsigned int)v5;
            }
            goto LABEL_9;
          }
          goto LABEL_36;
        }
LABEL_43:
        if ( *(char *)(v4 + 23) < 0 )
        {
          v30 = sub_1648A40(v4);
          v32 = (_DWORD *)(v30 + v31);
          v33 = *(char *)(v4 + 23) >= 0 ? 0LL : (_DWORD *)sub_1648A40(v4);
          if ( v33 != v32 )
          {
            while ( 1 )
            {
              v34 = v33[2];
              if ( v34 <= v14 && v33[3] > v14 )
                break;
              v33 += 4;
              if ( v32 == v33 )
                goto LABEL_58;
            }
            if ( !*(_DWORD *)(*(_QWORD *)v33 + 8LL) )
            {
              LOBYTE(v5) = v3 == 22 || v3 == 37;
              if ( (_BYTE)v5 )
                LOBYTE(v5) = *(_BYTE *)(**(_QWORD **)(v4
                                                    + 24
                                                    * (v34
                                                     - (unsigned __int64)(*(_DWORD *)(v4 + 20) & 0xFFFFFFF)
                                                     + v14
                                                     - v34))
                                      + 8LL) == 15;
            }
            return (unsigned int)v5;
          }
        }
LABEL_58:
        BUG();
      }
    }
    v14 = a2 - 1;
    if ( (unsigned int)a2 < v7 - 2 )
      goto LABEL_16;
    goto LABEL_43;
  }
  if ( !(_DWORD)a2 )
  {
    if ( !(unsigned __int8)sub_1560260(v4 + 56, a2, a3) )
    {
      v6 = *(_QWORD *)(v4 - 24);
      if ( !*(_BYTE *)(v6 + 16) )
      {
LABEL_5:
        v35[0] = *(_QWORD *)(v6 + 112);
        LODWORD(v5) = sub_1560260(v35, 0, v3);
        return (unsigned int)v5;
      }
LABEL_9:
      LODWORD(v5) = 0;
      return (unsigned int)v5;
    }
    goto LABEL_36;
  }
  v16 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
  if ( *(char *)(v4 + 23) >= 0 )
    goto LABEL_25;
  v17 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
  v19 = v17 + v18;
  if ( *(char *)(v4 + 23) >= 0 )
  {
    if ( !(unsigned int)(v19 >> 4) )
      goto LABEL_25;
LABEL_55:
    BUG();
  }
  if ( !(unsigned int)((v19 - sub_1648A40(v4)) >> 4) )
    goto LABEL_25;
  if ( *(char *)(v4 + 23) >= 0 )
    goto LABEL_55;
  v20 = *(_DWORD *)(sub_1648A40(v4) + 8);
  if ( *(char *)(v4 + 23) >= 0 )
    goto LABEL_58;
  v21 = sub_1648A40(v4);
  v16 += v20 - *(_DWORD *)(v21 + v22 - 4);
LABEL_25:
  v23 = a2 - 1;
  if ( (unsigned int)a2 < v16 )
  {
    if ( !(unsigned __int8)sub_1560290(v4 + 56, v23, v3) )
    {
      v29 = *(_QWORD *)(v4 - 24);
      if ( !*(_BYTE *)(v29 + 16) )
      {
        v35[0] = *(_QWORD *)(v29 + 112);
        LODWORD(v5) = sub_1560290(v35, v23, v3);
        return (unsigned int)v5;
      }
      goto LABEL_9;
    }
    goto LABEL_36;
  }
  if ( *(char *)(v4 + 23) >= 0 )
    goto LABEL_58;
  v24 = sub_1648A40(v4);
  v5 = v24 + v25;
  if ( *(char *)(v4 + 23) >= 0 )
    v26 = 0;
  else
    v26 = (_DWORD *)sub_1648A40(v4);
  while ( 1 )
  {
    if ( v26 == (_DWORD *)v5 )
      goto LABEL_58;
    v27 = v26[2];
    if ( v27 <= v23 && v26[3] > v23 )
      break;
    v26 += 4;
  }
  if ( *(_DWORD *)(*(_QWORD *)v26 + 8LL) )
    goto LABEL_9;
  LOBYTE(v5) = v3 == 22 || v3 == 37;
  if ( (_BYTE)v5 )
    LOBYTE(v5) = *(_BYTE *)(**(_QWORD **)(v4
                                        + 24 * (v27 - (unsigned __int64)(*(_DWORD *)(v4 + 20) & 0xFFFFFFF) + v23 - v27))
                          + 8LL) == 15;
  return (unsigned int)v5;
}

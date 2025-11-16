// Function: sub_11F2A60
// Address: 0x11f2a60
//
__int64 __fastcall sub_11F2A60(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v5; // r15d
  int *v7; // rax
  unsigned int v8; // r14d
  int v9; // eax
  _DWORD *v10; // rax
  __int64 v11; // r15
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // eax
  _DWORD *v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdx
  _DWORD *v21; // rax
  int v22; // r14d
  unsigned int v23; // eax
  _DWORD *v24; // rax
  __int64 v25; // r15
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 j; // r8
  char v30; // al
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 i; // r15
  char v34; // al
  __int64 v35; // rax
  __int64 v36; // rdx
  unsigned int v37; // eax
  __int64 k; // r15
  char v39; // al
  __int64 v40; // rax
  __int64 v41; // rdx
  unsigned int v42; // eax
  unsigned __int64 v43; // [rsp+8h] [rbp-78h]
  int v44; // [rsp+14h] [rbp-6Ch]
  int v45; // [rsp+14h] [rbp-6Ch]
  __int64 v46; // [rsp+18h] [rbp-68h]
  unsigned __int64 v47; // [rsp+18h] [rbp-68h]
  unsigned __int64 v48; // [rsp+18h] [rbp-68h]
  __int64 v49; // [rsp+20h] [rbp-60h]
  __int64 v50; // [rsp+20h] [rbp-60h]
  __int64 v51; // [rsp+20h] [rbp-60h]
  __int64 v52; // [rsp+28h] [rbp-58h]
  __int64 v53; // [rsp+28h] [rbp-58h]
  __int64 v54; // [rsp+28h] [rbp-58h]
  unsigned __int64 v55; // [rsp+30h] [rbp-50h] BYREF
  __int64 v56; // [rsp+38h] [rbp-48h]
  char v57; // [rsp+40h] [rbp-40h]

  if ( (unsigned __int8)sub_B2D610(a1, 47) )
    return 1;
  v5 = sub_B2D610(a1, 18);
  if ( (_BYTE)v5 )
    return 1;
  if ( !a2 )
    return v5;
  if ( !a3 )
    return v5;
  v7 = *(int **)(a2 + 8);
  if ( !v7 )
    return v5;
  if ( LOBYTE(qword_4F91668[8]) )
    return 1;
  v8 = LOBYTE(qword_4F91BA8[8]);
  if ( !LOBYTE(qword_4F91BA8[8]) )
    return v5;
  if ( LOBYTE(qword_4F919E8[8]) )
    goto LABEL_27;
  v9 = *v7;
  if ( !v9 )
  {
    if ( !LOBYTE(qword_4F91908[8]) )
      goto LABEL_13;
LABEL_27:
    if ( !a1 )
      return v5;
LABEL_28:
    sub_B2EE70((__int64)&v55, a1, 0);
    if ( v57 )
    {
      LOBYTE(v15) = sub_D84450(a2, v55);
      v5 = v15;
      if ( !(_BYTE)v15 )
        return v5;
    }
    v16 = *(_DWORD **)(a2 + 8);
    v52 = a1 + 72;
    if ( v16 )
    {
      if ( *v16 == 2 )
      {
        v47 = 0;
        v50 = *(_QWORD *)(a1 + 80);
        if ( v50 != a1 + 72 )
        {
          do
          {
            if ( !v50 )
              BUG();
            for ( i = *(_QWORD *)(v50 + 32); v50 + 24 != i; i = *(_QWORD *)(i + 8) )
            {
              if ( !i )
                BUG();
              v34 = *(_BYTE *)(i - 24);
              if ( v34 == 34 || v34 == 85 )
              {
                v35 = sub_D84370(a2, i - 24, 0, 0);
                v56 = v36;
                v55 = v35;
                if ( (_BYTE)v36 )
                  v47 += v55;
              }
            }
            v50 = *(_QWORD *)(v50 + 8);
          }
          while ( v50 != v52 );
        }
        LOBYTE(v37) = sub_D84450(a2, v47);
        v5 = v37;
        if ( !(_BYTE)v37 )
          return v5;
      }
    }
    v17 = *(_QWORD *)(a1 + 80);
    if ( v17 != v52 )
    {
      while ( 1 )
      {
        v18 = v17 - 24;
        if ( !v17 )
          v18 = 0;
        v19 = sub_FDD2C0(a3, v18, 0);
        v56 = v20;
        v55 = v19;
        if ( !(_BYTE)v20 || !sub_D84450(a2, v55) )
          break;
        v17 = *(_QWORD *)(v17 + 8);
        if ( v17 == v52 )
          return 1;
      }
      return 0;
    }
    return 1;
  }
  if ( v9 != 2 )
  {
LABEL_13:
    if ( !LOBYTE(qword_4F91AC8[8]) )
      goto LABEL_14;
LABEL_65:
    if ( !(unsigned __int8)sub_D84430(a2) )
      goto LABEL_46;
    goto LABEL_50;
  }
  if ( !(unsigned __int8)sub_D845F0(a2) && LOBYTE(qword_4F91828[8])
    || (unsigned __int8)sub_D845F0(a2) && LOBYTE(qword_4F91748[8]) )
  {
LABEL_46:
    if ( !a1 || !*(_QWORD *)(a2 + 8) )
      return v5;
    goto LABEL_28;
  }
  if ( LOBYTE(qword_4F91AC8[8]) )
    goto LABEL_65;
LABEL_50:
  v21 = *(_DWORD **)(a2 + 8);
  if ( !v21 )
    return v8;
  if ( *v21 != 2 )
  {
LABEL_14:
    v44 = qword_4F91588[8];
    if ( a1 )
    {
      sub_B2EE70((__int64)&v55, a1, 0);
      if ( v57 && sub_D85370(a2, v44, v55) )
        return 0;
      v10 = *(_DWORD **)(a2 + 8);
      v49 = a1 + 72;
      if ( v10 )
      {
        if ( *v10 == 2 )
        {
          v43 = 0;
          v46 = *(_QWORD *)(a1 + 80);
          if ( a1 + 72 != v46 )
          {
            do
            {
              if ( !v46 )
                BUG();
              for ( j = *(_QWORD *)(v46 + 32); v46 + 24 != j; j = *(_QWORD *)(j + 8) )
              {
                if ( !j )
                  BUG();
                v30 = *(_BYTE *)(j - 24);
                if ( v30 == 34 || v30 == 85 )
                {
                  v54 = j;
                  v31 = sub_D84370(a2, j - 24, 0, 0);
                  j = v54;
                  v56 = v32;
                  v55 = v31;
                  if ( (_BYTE)v32 )
                    v43 += v55;
                }
              }
              v46 = *(_QWORD *)(v46 + 8);
            }
            while ( v49 != v46 );
          }
          if ( sub_D85370(a2, v44, v43) )
            return 0;
        }
      }
      v11 = *(_QWORD *)(a1 + 80);
      if ( v49 != v11 )
      {
        while ( 1 )
        {
          v12 = v11 - 24;
          if ( !v11 )
            v12 = 0;
          v13 = sub_FDD2C0(a3, v12, 0);
          v56 = v14;
          v55 = v13;
          if ( (_BYTE)v14 )
          {
            if ( sub_D85370(a2, v44, v55) )
              break;
          }
          v11 = *(_QWORD *)(v11 + 8);
          if ( v49 == v11 )
            return v8;
        }
        return 0;
      }
    }
    return v8;
  }
  if ( a1 )
  {
    v45 = qword_4F914A8[8];
    v22 = qword_4F914A8[8];
    sub_B2EE70((__int64)&v55, a1, 0);
    if ( !v57 || (LOBYTE(v23) = sub_D853A0(a2, v22, v55), v5 = v23, (_BYTE)v23) )
    {
      v24 = *(_DWORD **)(a2 + 8);
      v53 = a1 + 72;
      if ( !v24 || *v24 != 2 )
        goto LABEL_57;
      v48 = 0;
      v51 = *(_QWORD *)(a1 + 80);
      if ( a1 + 72 != v51 )
      {
        do
        {
          if ( !v51 )
            BUG();
          for ( k = *(_QWORD *)(v51 + 32); v51 + 24 != k; k = *(_QWORD *)(k + 8) )
          {
            if ( !k )
              BUG();
            v39 = *(_BYTE *)(k - 24);
            if ( v39 == 34 || v39 == 85 )
            {
              v40 = sub_D84370(a2, k - 24, 0, 0);
              v56 = v41;
              v55 = v40;
              if ( (_BYTE)v41 )
                v48 += v55;
            }
          }
          v51 = *(_QWORD *)(v51 + 8);
        }
        while ( v51 != v53 );
      }
      LOBYTE(v42) = sub_D853A0(a2, v45, v48);
      v5 = v42;
      if ( (_BYTE)v42 )
      {
LABEL_57:
        v25 = *(_QWORD *)(a1 + 80);
        if ( v25 != v53 )
        {
          while ( 1 )
          {
            v26 = v25 - 24;
            if ( !v25 )
              v26 = 0;
            v27 = sub_FDD2C0(a3, v26, 0);
            v56 = v28;
            v55 = v27;
            if ( !(_BYTE)v28 || !sub_D853A0(a2, v45, v55) )
              break;
            v25 = *(_QWORD *)(v25 + 8);
            if ( v25 == v53 )
              return 1;
          }
          return 0;
        }
        return 1;
      }
    }
  }
  return v5;
}

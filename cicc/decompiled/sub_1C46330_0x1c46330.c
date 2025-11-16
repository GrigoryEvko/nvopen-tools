// Function: sub_1C46330
// Address: 0x1c46330
//
__int64 __fastcall sub_1C46330(_QWORD *a1, __int64 a2, char a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  _QWORD *v5; // r15
  _QWORD *v6; // rbx
  _QWORD *v7; // r13
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  bool v10; // al
  __int64 v11; // rdi
  _QWORD *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  _QWORD *v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *i; // rbx
  bool v22; // zf
  __int64 v23; // rdi
  __int64 v25; // [rsp+8h] [rbp-78h]
  _QWORD *v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  unsigned __int64 v28; // [rsp+20h] [rbp-60h]
  char v30; // [rsp+3Dh] [rbp-43h] BYREF
  char v31; // [rsp+3Eh] [rbp-42h] BYREF
  char v32; // [rsp+3Fh] [rbp-41h] BYREF
  unsigned __int64 v33; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 *v34[7]; // [rsp+48h] [rbp-38h] BYREF

  result = *(_QWORD *)(a2 + 80);
  v27 = result;
  v25 = a2 + 72;
  v26 = a1 + 4;
  while ( v25 != result )
  {
    v30 = 0;
    v31 = 0;
    v4 = v27 - 24;
    if ( !v27 )
      v4 = 0;
    v33 = v4;
    v5 = *(_QWORD **)(v4 + 48);
    v6 = (_QWORD *)(v4 + 40);
    if ( a3 )
    {
      v7 = (_QWORD *)(v4 + 40);
      if ( v5 != v6 )
      {
        while ( 1 )
        {
          v8 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
          v7 = (_QWORD *)v8;
          if ( !v8 )
            BUG();
          if ( *(_BYTE *)(v8 - 8) == 78 )
          {
            v9 = *(_QWORD *)(v8 - 48);
            if ( !*(_BYTE *)(v9 + 16) && (*(_BYTE *)(v9 + 33) & 0x20) != 0 )
            {
              v28 = v8;
              v10 = sub_1C301F0(*(_DWORD *)(v9 + 36));
              v8 = v28;
              if ( v10 )
                break;
            }
          }
          if ( (_QWORD *)v8 == v5 )
            goto LABEL_35;
        }
        for ( ; v7 != v6; v31 |= LOBYTE(v34[0]) )
        {
          v11 = (__int64)(v7 - 3);
          if ( !v7 )
            v11 = 0;
          v32 = 0;
          LOBYTE(v34[0]) = 0;
          sub_1C45690(v11, &v32, v34);
          v7 = (_QWORD *)v7[1];
          v30 |= v32;
        }
        goto LABEL_17;
      }
    }
    else if ( v5 != v6 )
    {
      while ( 1 )
      {
        if ( !v5 )
          BUG();
        if ( *((_BYTE *)v5 - 8) == 78 )
        {
          v20 = *(v5 - 6);
          if ( !*(_BYTE *)(v20 + 16) && (*(_BYTE *)(v20 + 33) & 0x20) != 0 && sub_1C301F0(*(_DWORD *)(v20 + 36)) )
            break;
        }
        v5 = (_QWORD *)v5[1];
        if ( v5 == v6 )
          goto LABEL_35;
      }
      for ( i = *(_QWORD **)(v4 + 48); i != v5; v31 |= LOBYTE(v34[0]) )
      {
        v32 = 0;
        LOBYTE(v34[0]) = 0;
        v22 = (*v5 & 0xFFFFFFFFFFFFFFF8LL) == 0;
        v23 = (*v5 & 0xFFFFFFFFFFFFFFF8LL) - 24;
        v5 = (_QWORD *)(*v5 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v22 )
          v23 = 0;
        sub_1C45690(v23, &v32, v34);
        v30 |= v32;
      }
LABEL_17:
      v12 = (_QWORD *)a1[5];
      if ( !v12 )
        goto LABEL_36;
      goto LABEL_18;
    }
LABEL_35:
    sub_1C458C0(v4, &v30, &v31);
    v12 = (_QWORD *)a1[5];
    if ( !v12 )
    {
LABEL_36:
      v13 = (__int64)v26;
LABEL_24:
      v34[0] = &v33;
      v13 = sub_1C46280(a1 + 3, (_QWORD *)v13, v34);
      goto LABEL_25;
    }
LABEL_18:
    v13 = (__int64)v26;
    do
    {
      while ( 1 )
      {
        v14 = v12[2];
        v15 = v12[3];
        if ( v12[4] >= v33 )
          break;
        v12 = (_QWORD *)v12[3];
        if ( !v15 )
          goto LABEL_22;
      }
      v13 = (__int64)v12;
      v12 = (_QWORD *)v12[2];
    }
    while ( v14 );
LABEL_22:
    if ( v26 == (_QWORD *)v13 || *(_QWORD *)(v13 + 32) > v33 )
      goto LABEL_24;
LABEL_25:
    *(_BYTE *)(v13 + 40) = v30;
    v16 = (_QWORD *)a1[11];
    if ( v16 )
    {
      v17 = (__int64)(a1 + 10);
      do
      {
        while ( 1 )
        {
          v18 = v16[2];
          v19 = v16[3];
          if ( v16[4] >= v33 )
            break;
          v16 = (_QWORD *)v16[3];
          if ( !v19 )
            goto LABEL_30;
        }
        v17 = (__int64)v16;
        v16 = (_QWORD *)v16[2];
      }
      while ( v18 );
LABEL_30:
      if ( a1 + 10 != (_QWORD *)v17 && *(_QWORD *)(v17 + 32) <= v33 )
        goto LABEL_33;
    }
    else
    {
      v17 = (__int64)(a1 + 10);
    }
    v34[0] = &v33;
    v17 = sub_1C46280(a1 + 9, (_QWORD *)v17, v34);
LABEL_33:
    *(_BYTE *)(v17 + 40) = v31;
    result = *(_QWORD *)(v27 + 8);
    v27 = result;
  }
  return result;
}

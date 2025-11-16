// Function: sub_14EEBA0
// Address: 0x14eeba0
//
__int64 __fastcall sub_14EEBA0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v5; // r10d
  unsigned __int64 v6; // r8
  unsigned __int64 v7; // r11
  const char *v8; // rax
  __int64 v9; // rax
  unsigned int v11; // r14d
  __int64 v12; // rsi
  unsigned int v13; // r15d
  unsigned __int64 v14; // r9
  _QWORD *v15; // rbx
  unsigned int v16; // r8d
  unsigned __int64 v17; // r9
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rdx
  unsigned int v21; // r8d
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdx
  char v26; // cl
  unsigned int v27; // edi
  unsigned __int64 v28; // r11
  unsigned __int64 v29; // r8
  unsigned __int64 v30; // r14
  unsigned int v31; // ebx
  unsigned __int64 *v32; // r10
  unsigned __int64 v33; // rsi
  unsigned int v34; // r8d
  unsigned __int64 v35; // r14
  __int64 v36; // rbx
  unsigned __int64 v37; // r11
  unsigned __int64 v38; // r8
  unsigned int v39; // r14d
  unsigned __int64 *v40; // r10
  unsigned __int64 v41; // rsi
  unsigned int v42; // r8d
  unsigned __int64 v43; // rbx
  unsigned int v44; // edi
  unsigned __int64 v45; // rax
  unsigned int v46; // r8d
  __int64 v47; // rax
  __int64 v48; // rdx
  char v49; // cl
  unsigned __int64 v50; // r9
  unsigned int v51; // r8d
  __int64 v52; // rax
  __int64 v53; // rdx
  char v54; // cl
  unsigned __int64 v55; // r9
  __int64 v56; // [rsp+0h] [rbp-70h]
  __int64 v57; // [rsp+8h] [rbp-68h]
  __int64 v58; // [rsp+10h] [rbp-60h] BYREF
  __int64 v59; // [rsp+18h] [rbp-58h]
  unsigned __int64 v60[2]; // [rsp+20h] [rbp-50h] BYREF
  char v61; // [rsp+30h] [rbp-40h] BYREF
  char v62; // [rsp+31h] [rbp-3Fh]

  v56 = 0;
  v57 = 0;
  if ( (unsigned __int8)sub_15127D0(a2, a3, 0) )
  {
    v62 = 1;
    v8 = "Invalid record";
LABEL_5:
    v60[0] = (unsigned __int64)v8;
    v61 = 3;
    sub_14EE0F0(&v58, (__int64)v60);
    v9 = v58;
    *(_BYTE *)(a1 + 16) |= 3u;
    *(_QWORD *)a1 = v9 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  while ( 1 )
  {
    v5 = *(_DWORD *)(a2 + 32);
    if ( v5 )
    {
      v11 = *(_DWORD *)(a2 + 36);
      if ( v5 >= v11 )
        goto LABEL_22;
      v6 = *(_QWORD *)(a2 + 8);
      v7 = *(_QWORD *)(a2 + 16);
      v12 = *(_QWORD *)(a2 + 24);
      v13 = v11 - v5;
      if ( v7 >= v6 )
        goto LABEL_54;
    }
    else
    {
      v6 = *(_QWORD *)(a2 + 8);
      v7 = *(_QWORD *)(a2 + 16);
      if ( v6 <= v7 )
        goto LABEL_4;
      v13 = *(_DWORD *)(a2 + 36);
      if ( !v13 )
      {
        v11 = 0;
LABEL_22:
        v20 = *(_QWORD *)(a2 + 24);
        *(_DWORD *)(a2 + 32) = v5 - v11;
        v19 = v20 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v11));
        v18 = v20 >> v11;
        *(_QWORD *)(a2 + 24) = v18;
        goto LABEL_13;
      }
      v11 = *(_DWORD *)(a2 + 36);
      v12 = 0;
    }
    v14 = v7 + 8;
    v15 = (_QWORD *)(v7 + *(_QWORD *)a2);
    if ( v7 + 8 > v6 )
    {
      *(_QWORD *)(a2 + 24) = 0;
      v21 = v6 - v7;
      if ( !v21 )
        goto LABEL_64;
      v22 = v21;
      v23 = 0;
      v24 = 0;
      do
      {
        v25 = *((unsigned __int8 *)v15 + v23);
        v26 = 8 * v23++;
        v24 |= v25 << v26;
        *(_QWORD *)(a2 + 24) = v24;
      }
      while ( v21 != v23 );
      v16 = 8 * v21;
      v14 = v7 + v22;
    }
    else
    {
      v16 = 64;
      *(_QWORD *)(a2 + 24) = *v15;
    }
    *(_QWORD *)(a2 + 16) = v14;
    *(_DWORD *)(a2 + 32) = v16;
    if ( v16 < v13 )
      goto LABEL_54;
    v17 = *(_QWORD *)(a2 + 24);
    v18 = v17 >> v13;
    *(_QWORD *)(a2 + 24) = v17 >> v13;
    *(_DWORD *)(a2 + 32) = v5 - v11 + v16;
    v19 = ((v17 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v5 - (unsigned __int8)v11 + 64))) << v5) | v12;
LABEL_13:
    if ( !(_DWORD)v19 )
      break;
    if ( (_DWORD)v19 == 1 )
    {
      v27 = *(_DWORD *)(a2 + 32);
      if ( v27 > 7 )
      {
        LOBYTE(v35) = v18;
        *(_QWORD *)(a2 + 24) = v18 >> 8;
        *(_DWORD *)(a2 + 32) = v27 - 8;
      }
      else
      {
        v28 = *(_QWORD *)(a2 + 16);
        v29 = *(_QWORD *)(a2 + 8);
        v30 = 0;
        if ( v27 )
          v30 = v18;
        v31 = 8 - v27;
        if ( v28 >= v29 )
          goto LABEL_54;
        v32 = (unsigned __int64 *)(v28 + *(_QWORD *)a2);
        if ( v29 < v28 + 8 )
        {
          *(_QWORD *)(a2 + 24) = 0;
          v51 = v29 - v28;
          if ( !v51 )
          {
LABEL_64:
            *(_DWORD *)(a2 + 32) = 0;
LABEL_54:
            sub_16BD130("Unexpected end of file", 1);
          }
          v52 = 0;
          v33 = 0;
          do
          {
            v53 = *((unsigned __int8 *)v32 + v52);
            v54 = 8 * v52++;
            v33 |= v53 << v54;
            *(_QWORD *)(a2 + 24) = v33;
          }
          while ( v51 != v52 );
          v55 = v28 + v51;
          v34 = 8 * v51;
          *(_QWORD *)(a2 + 16) = v55;
          *(_DWORD *)(a2 + 32) = v34;
          if ( v31 > v34 )
            goto LABEL_54;
        }
        else
        {
          v33 = *v32;
          *(_QWORD *)(a2 + 16) = v28 + 8;
          v34 = 64;
        }
        *(_QWORD *)(a2 + 24) = v33 >> v31;
        *(_DWORD *)(a2 + 32) = v27 + v34 - 8;
        v35 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v27 + 56)) & v33) << v27) | v30;
      }
      if ( (v35 & 0x80) != 0 )
      {
        do
        {
          v44 = *(_DWORD *)(a2 + 32);
          if ( v44 <= 7 )
          {
            v36 = 0;
            if ( v44 )
              v36 = *(_QWORD *)(a2 + 24);
            v37 = *(_QWORD *)(a2 + 16);
            v38 = *(_QWORD *)(a2 + 8);
            v39 = 8 - v44;
            if ( v37 >= v38 )
              goto LABEL_54;
            v40 = (unsigned __int64 *)(v37 + *(_QWORD *)a2);
            if ( v38 < v37 + 8 )
            {
              *(_QWORD *)(a2 + 24) = 0;
              v46 = v38 - v37;
              if ( !v46 )
                goto LABEL_64;
              v47 = 0;
              v41 = 0;
              do
              {
                v48 = *((unsigned __int8 *)v40 + v47);
                v49 = 8 * v47++;
                v41 |= v48 << v49;
                *(_QWORD *)(a2 + 24) = v41;
              }
              while ( v46 != v47 );
              v50 = v37 + v46;
              v42 = 8 * v46;
              *(_QWORD *)(a2 + 16) = v50;
              *(_DWORD *)(a2 + 32) = v42;
              if ( v39 > v42 )
                goto LABEL_54;
            }
            else
            {
              v41 = *v40;
              *(_QWORD *)(a2 + 16) = v37 + 8;
              v42 = 64;
            }
            *(_QWORD *)(a2 + 24) = v41 >> v39;
            *(_DWORD *)(a2 + 32) = v42 + v44 - 8;
            v43 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v44 + 56)) & v41) << v44) | v36;
          }
          else
          {
            v45 = *(_QWORD *)(a2 + 24);
            *(_DWORD *)(a2 + 32) = v44 - 8;
            LOBYTE(v43) = v45;
            *(_QWORD *)(a2 + 24) = v45 >> 8;
          }
        }
        while ( (v43 & 0x80) != 0 );
      }
      if ( (unsigned __int8)sub_14ED8F0(a2) )
        goto LABEL_4;
    }
    else if ( (_DWORD)v19 == 2 )
    {
      sub_1513230(a2);
    }
    else
    {
      v58 = 0;
      v59 = 0;
      v60[0] = (unsigned __int64)&v61;
      v60[1] = 0x100000000LL;
      if ( (unsigned int)sub_1510D70(a2, v19, v60, &v58) == 1 )
      {
        v57 = v58;
        v56 = v59;
      }
      if ( (char *)v60[0] != &v61 )
        _libc_free(v60[0]);
    }
  }
  if ( !*(_DWORD *)(a2 + 72) || (unsigned __int8)sub_14EB5C0(a2) )
  {
LABEL_4:
    v62 = 1;
    v8 = "Malformed block";
    goto LABEL_5;
  }
  *(_BYTE *)(a1 + 16) = *(_BYTE *)(a1 + 16) & 0xFC | 2;
  *(_QWORD *)a1 = v57;
  *(_QWORD *)(a1 + 8) = v56;
  return a1;
}

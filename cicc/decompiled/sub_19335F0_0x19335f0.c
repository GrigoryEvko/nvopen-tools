// Function: sub_19335F0
// Address: 0x19335f0
//
__int64 __fastcall sub_19335F0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned __int64 v4; // rax
  unsigned int v5; // edx
  char v6; // cl
  __int64 v7; // r14
  __int64 *v8; // rax
  int v9; // eax
  __int64 v10; // r13
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // r9
  __int64 v14; // rax
  __int64 v15; // r13
  unsigned __int64 *v16; // r15
  unsigned __int64 v17; // rax
  unsigned __int64 *v18; // r13
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  unsigned __int64 *v21; // rax
  unsigned __int64 *v22; // rsi
  int v23; // eax
  int v24; // edx
  __int64 v26; // r14
  char v27; // al
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r15
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int64 *v48; // rsi
  __int64 v49; // rax
  __int64 v50; // [rsp+0h] [rbp-60h]
  __int64 i; // [rsp+10h] [rbp-50h]
  _QWORD v52[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = sub_145CBF0((__int64 *)(a1 + 96), 56, 16);
  LODWORD(v4) = sub_1648EF0(a2);
  *(_QWORD *)(v3 + 16) = 0;
  *(_QWORD *)(v3 + 8) = 0xFFFFFFFD00000006LL;
  *(_DWORD *)(v3 + 32) = v4;
  v4 = (unsigned int)v4;
  *(_QWORD *)(v3 + 24) = 0;
  *(_QWORD *)v3 = off_49F34F8;
  *(_QWORD *)(v3 + 36) = 0;
  *(_QWORD *)(v3 + 44) = 0xFFFFFFFF00000000LL;
  *(_BYTE *)(v3 + 52) = 0;
  if ( !(_DWORD)v4 )
  {
    v6 = 0;
    goto LABEL_6;
  }
  v4 = (unsigned int)v4 - 1LL;
  if ( !v4 )
  {
    v6 = 0;
LABEL_6:
    if ( !*(_DWORD *)(a1 + 208) )
      goto LABEL_4;
    goto LABEL_7;
  }
  _BitScanReverse64(&v4, v4);
  v5 = 64 - (v4 ^ 0x3F);
  v6 = 64 - (v4 ^ 0x3F);
  v4 = v5;
  if ( *(_DWORD *)(a1 + 208) <= v5 )
  {
LABEL_4:
    v7 = sub_145CBF0((__int64 *)(a1 + 96), 8LL << v6, 8);
    goto LABEL_9;
  }
LABEL_7:
  v8 = (__int64 *)(*(_QWORD *)(a1 + 200) + 8 * v4);
  v7 = *v8;
  if ( !*v8 )
    goto LABEL_4;
  *v8 = *(_QWORD *)v7;
LABEL_9:
  v9 = *(unsigned __int8 *)(a2 + 16);
  v10 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(v3 + 24) = v7;
  *(_DWORD *)(v3 + 12) = v9 - 24;
  *(_QWORD *)(v3 + 40) = *(_QWORD *)a2;
  if ( v10 )
  {
    do
    {
      v11 = sub_1648700(v10);
      v12 = *(_QWORD *)(v3 + 24);
      v13 = v11;
      v14 = *(unsigned int *)(v3 + 36);
      *(_DWORD *)(v3 + 36) = v14 + 1;
      *(_QWORD *)(v12 + 8 * v14) = v13;
      v10 = *(_QWORD *)(v10 + 8);
    }
    while ( v10 );
    v7 = *(_QWORD *)(v3 + 24);
  }
  v15 = 8LL * *(unsigned int *)(v3 + 36);
  v16 = (unsigned __int64 *)(v7 + v15);
  if ( v7 + v15 != v7 )
  {
    _BitScanReverse64(&v17, v15 >> 3);
    sub_192DED0((char *)v7, (unsigned __int64 *)(v7 + v15), 2LL * (int)(63 - (v17 ^ 0x3F)));
    if ( (unsigned __int64)v15 <= 0x80 )
    {
      sub_192D700((unsigned __int64 *)v7, (unsigned __int64 *)(v7 + v15));
      if ( !(unsigned __int8)sub_19306C0(a2) )
        goto LABEL_19;
      goto LABEL_23;
    }
    v18 = (unsigned __int64 *)(v7 + 128);
    sub_192D700((unsigned __int64 *)v7, (unsigned __int64 *)(v7 + 128));
    if ( v16 != (unsigned __int64 *)(v7 + 128) )
    {
      do
      {
        while ( 1 )
        {
          v19 = *v18;
          v20 = *(v18 - 1);
          v21 = v18 - 1;
          if ( *v18 < v20 )
            break;
          v48 = v18++;
          *v48 = v19;
          if ( v16 == v18 )
            goto LABEL_18;
        }
        do
        {
          v21[1] = v20;
          v22 = v21;
          v20 = *--v21;
        }
        while ( v19 < v20 );
        ++v18;
        *v22 = v19;
      }
      while ( v16 != v18 );
    }
  }
LABEL_18:
  if ( !(unsigned __int8)sub_19306C0(a2) )
    goto LABEL_19;
LABEL_23:
  v26 = *(_QWORD *)(a2 + 32);
  for ( i = *(_QWORD *)(a2 + 40) + 40LL; i != v26; v26 = *(_QWORD *)(v26 + 8) )
  {
    if ( !v26 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v26 - 8) - 25 <= 9 )
      break;
    if ( (unsigned __int8)sub_19306C0(v26 - 24) )
    {
      v27 = *(_BYTE *)(v26 - 8);
      if ( v27 != 54 )
      {
        if ( v27 == 78 )
        {
          if ( (unsigned __int8)sub_1560260((_QWORD *)(v26 + 32), -1, 36) )
            continue;
          if ( *(char *)(v26 - 1) >= 0 )
            goto LABEL_76;
          v29 = sub_1648A40(v26 - 24);
          v31 = v29 + v30;
          v32 = 0;
          if ( *(char *)(v26 - 1) < 0 )
            v32 = sub_1648A40(v26 - 24);
          if ( !(unsigned int)((v31 - v32) >> 4) )
          {
LABEL_76:
            v33 = *(_QWORD *)(v26 - 48);
            if ( !*(_BYTE *)(v33 + 16) )
            {
              v52[0] = *(_QWORD *)(v33 + 112);
              if ( (unsigned __int8)sub_1560260(v52, -1, 36) )
                continue;
            }
          }
          if ( (unsigned __int8)sub_1560260((_QWORD *)(v26 + 32), -1, 37) )
            continue;
          if ( *(char *)(v26 - 1) >= 0
            || ((v34 = sub_1648A40(v26 - 24), v36 = v34 + v35, *(char *)(v26 - 1) >= 0)
              ? (v37 = 0)
              : (v37 = sub_1648A40(v26 - 24)),
                v37 == v36) )
          {
LABEL_68:
            v49 = *(_QWORD *)(v26 - 48);
            if ( !*(_BYTE *)(v49 + 16) )
            {
              v52[0] = *(_QWORD *)(v49 + 112);
              if ( (unsigned __int8)sub_1560260(v52, -1, 37) )
                continue;
            }
          }
          else
          {
            while ( *(_DWORD *)(*(_QWORD *)v37 + 8LL) <= 1u )
            {
              v37 += 16;
              if ( v36 == v37 )
                goto LABEL_68;
            }
          }
          v27 = *(_BYTE *)(v26 - 8);
        }
        if ( v27 != 29 )
          goto LABEL_62;
        if ( !(unsigned __int8)sub_1560260((_QWORD *)(v26 + 32), -1, 36) )
        {
          if ( *(char *)(v26 - 1) < 0 )
          {
            v38 = sub_1648A40(v26 - 24);
            v40 = v39 + v38;
            v41 = 0;
            v50 = v40;
            if ( *(char *)(v26 - 1) < 0 )
              v41 = sub_1648A40(v26 - 24);
            if ( (unsigned int)((v50 - v41) >> 4) )
              goto LABEL_77;
          }
          v42 = *(_QWORD *)(v26 - 96);
          if ( *(_BYTE *)(v42 + 16) || (v52[0] = *(_QWORD *)(v42 + 112), !(unsigned __int8)sub_1560260(v52, -1, 36)) )
          {
LABEL_77:
            if ( !(unsigned __int8)sub_1560260((_QWORD *)(v26 + 32), -1, 37) )
            {
              if ( *(char *)(v26 - 1) < 0 )
              {
                v43 = sub_1648A40(v26 - 24);
                v45 = v43 + v44;
                v46 = *(char *)(v26 - 1) >= 0 ? 0LL : sub_1648A40(v26 - 24);
                if ( v46 != v45 )
                {
                  while ( *(_DWORD *)(*(_QWORD *)v46 + 8LL) <= 1u )
                  {
                    v46 += 16;
                    if ( v45 == v46 )
                      goto LABEL_60;
                  }
LABEL_62:
                  v28 = sub_1932D10(a1, v26 - 24);
                  goto LABEL_63;
                }
              }
LABEL_60:
              v47 = *(_QWORD *)(v26 - 96);
              if ( *(_BYTE *)(v47 + 16) )
                goto LABEL_62;
              v52[0] = *(_QWORD *)(v47 + 112);
              if ( !(unsigned __int8)sub_1560260(v52, -1, 37) )
                goto LABEL_62;
            }
          }
        }
      }
    }
  }
  v28 = 0;
LABEL_63:
  *(_DWORD *)(v3 + 48) = v28;
LABEL_19:
  v23 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)(v23 - 75) <= 1u )
  {
    v24 = *(unsigned __int16 *)(a2 + 18);
    BYTE1(v24) &= ~0x80u;
    *(_DWORD *)(v3 + 12) = v24 | ((v23 - 24) << 8);
  }
  return v3;
}

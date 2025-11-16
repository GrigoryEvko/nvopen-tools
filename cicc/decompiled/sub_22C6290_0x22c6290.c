// Function: sub_22C6290
// Address: 0x22c6290
//
char __fastcall sub_22C6290(unsigned __int8 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int8 **v13; // r14
  unsigned __int8 **v14; // r13
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rdi
  unsigned int v18; // r14d
  __int64 v19; // rdi
  unsigned int v20; // r14d
  __int64 v21; // rax
  int v22; // eax

  LODWORD(v2) = *a1;
  if ( (_BYTE)v2 == 61 || (_BYTE)v2 == 62 )
  {
    v4 = *((_QWORD *)a1 - 4);
    v2 = *(_QWORD *)(v4 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v2 + 8) - 17 <= 1 )
      v2 = **(_QWORD **)(v2 + 16);
    LODWORD(v2) = *(_DWORD *)(v2 + 8) >> 8;
    if ( !(_DWORD)v2 )
      LOBYTE(v2) = sub_22C6020((unsigned __int8 *)v4, a2, 1);
  }
  else
  {
    if ( (_BYTE)v2 != 85 )
    {
      if ( (unsigned __int8)(v2 - 34) > 0x33u )
        return v2;
      v5 = 0x8000000000041LL;
      if ( !_bittest64(&v5, (unsigned int)(v2 - 34)) )
        return v2;
      if ( (_DWORD)v2 == 40 )
      {
        v6 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
      }
      else
      {
        v6 = -96;
        if ( (_DWORD)v2 != 34 )
          goto LABEL_51;
      }
LABEL_13:
      if ( (a1[7] & 0x80u) == 0 )
        goto LABEL_19;
      v7 = sub_BD2BC0((__int64)a1);
      v9 = v7 + v8;
      if ( (a1[7] & 0x80u) == 0 )
      {
        if ( !(unsigned int)(v9 >> 4) )
        {
LABEL_19:
          v13 = (unsigned __int8 **)&a1[v6];
          v2 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
          v14 = (unsigned __int8 **)&a1[-v2];
          if ( &a1[-v2] != (unsigned __int8 *)v13 )
          {
            do
            {
              v2 = *((_QWORD *)*v14 + 1);
              if ( *(_BYTE *)(v2 + 8) == 14 )
              {
                LOBYTE(v2) = sub_B49C40(
                               (__int64)a1,
                               ((char *)v14 - (char *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)]) >> 5,
                               0);
                if ( (_BYTE)v2 )
                  LOBYTE(v2) = sub_22C6250(*v14, a2, 0);
              }
              v14 += 4;
            }
            while ( v13 != v14 );
          }
          return v2;
        }
      }
      else
      {
        if ( !(unsigned int)((v9 - sub_BD2BC0((__int64)a1)) >> 4) )
          goto LABEL_19;
        if ( (a1[7] & 0x80u) != 0 )
        {
          v10 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
          if ( (a1[7] & 0x80u) != 0 )
          {
            v11 = sub_BD2BC0((__int64)a1);
            v6 -= 32LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
            goto LABEL_19;
          }
LABEL_51:
          BUG();
        }
      }
      BUG();
    }
    v15 = *((_QWORD *)a1 - 4);
    v6 = -32;
    if ( !v15
      || *(_BYTE *)v15
      || *(_QWORD *)(v15 + 24) != *((_QWORD *)a1 + 10)
      || (*(_BYTE *)(v15 + 33) & 0x20) == 0
      || (unsigned int)(*(_DWORD *)(v15 + 36) - 238) > 7
      || ((1LL << (*(_BYTE *)(v15 + 36) + 18)) & 0xAD) == 0 )
    {
      goto LABEL_13;
    }
    v16 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
    v17 = *(_QWORD *)&a1[32 * (3 - v16)];
    v18 = *(_DWORD *)(v17 + 32);
    if ( v18 <= 0x40 )
      LOBYTE(v2) = *(_QWORD *)(v17 + 24) == 0;
    else
      LOBYTE(v2) = v18 == (unsigned int)sub_C444A0(v17 + 24);
    if ( (_BYTE)v2 )
    {
      v2 = 32 * (2 - v16);
      v19 = *(_QWORD *)&a1[v2];
      if ( *(_BYTE *)v19 == 17 )
      {
        v20 = *(_DWORD *)(v19 + 32);
        LOBYTE(v2) = v20 <= 0x40 ? *(_QWORD *)(v19 + 24) == 0 : v20 == (unsigned int)sub_C444A0(v19 + 24);
        if ( !(_BYTE)v2 )
        {
          sub_22C6250(*(unsigned __int8 **)&a1[-32 * v16], a2, 1);
          v21 = *((_QWORD *)a1 - 4);
          if ( !v21 || *(_BYTE *)v21 || *(_QWORD *)(v21 + 24) != *((_QWORD *)a1 + 10) )
            BUG();
          v22 = *(_DWORD *)(v21 + 36);
          if ( v22 == 238 || (LODWORD(v2) = v22 - 240, (unsigned int)v2 <= 1) )
            LOBYTE(v2) = sub_22C6250(*(unsigned __int8 **)&a1[32 * (1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))], a2, 1);
        }
      }
    }
  }
  return v2;
}

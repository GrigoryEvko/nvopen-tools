// Function: sub_DBA820
// Address: 0xdba820
//
__int64 __fastcall sub_DBA820(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  char *v3; // rax
  __int64 v4; // rdx
  char *v5; // rsi
  __int64 v6; // rdx
  char *v7; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // rdi

  v9 = sub_DB9E00(a1, a2);
  v10 = v9;
  v2 = *((unsigned __int8 *)v9 + 152);
  if ( (_BYTE)v2 )
  {
    v3 = (char *)*v9;
    v4 = 112LL * *((unsigned int *)v10 + 2);
    v5 = (char *)(*v10 + v4);
    v6 = 0x6DB6DB6DB6DB6DB7LL * (v4 >> 4);
    if ( v6 >> 2 )
    {
      v7 = &v3[448 * (v6 >> 2)];
      while ( !*((_DWORD *)v3 + 18) )
      {
        if ( *((_DWORD *)v3 + 46) )
        {
          LOBYTE(v2) = v5 == v3 + 112;
          return v2;
        }
        if ( *((_DWORD *)v3 + 74) )
        {
          LOBYTE(v2) = v5 == v3 + 224;
          return v2;
        }
        if ( *((_DWORD *)v3 + 102) )
        {
          LOBYTE(v2) = v5 == v3 + 336;
          return v2;
        }
        v3 += 448;
        if ( v3 == v7 )
        {
          v6 = 0x6DB6DB6DB6DB6DB7LL * ((v5 - v3) >> 4);
          goto LABEL_12;
        }
      }
      goto LABEL_9;
    }
LABEL_12:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          return v2;
        goto LABEL_15;
      }
      if ( *((_DWORD *)v3 + 18) )
      {
LABEL_9:
        LOBYTE(v2) = v5 == v3;
        return v2;
      }
      v3 += 112;
    }
    if ( !*((_DWORD *)v3 + 18) )
    {
      v3 += 112;
LABEL_15:
      if ( !*((_DWORD *)v3 + 18) )
        return v2;
      goto LABEL_9;
    }
    goto LABEL_9;
  }
  return v2;
}

// Function: sub_B55F20
// Address: 0xb55f20
//
__int64 __fastcall sub_B55F20(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // r13
  int v4; // r13d
  int v5; // eax
  int v6; // edx
  unsigned int v7; // r14d
  unsigned int v8; // eax
  __int64 v9; // rsi
  unsigned int v10; // r14d
  __int64 v11; // r13

  if ( *(char *)(a1 + 7) < 0 )
  {
    v1 = sub_BD2BC0(a1);
    v3 = v1 + v2;
    if ( *(char *)(a1 + 7) >= 0 )
    {
      if ( (unsigned int)(v3 >> 4) )
      {
        v6 = 0;
        v9 = 0;
        v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
        goto LABEL_7;
      }
    }
    else if ( (unsigned int)((v3 - sub_BD2BC0(a1)) >> 4) )
    {
      v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
      if ( *(char *)(a1 + 7) >= 0 )
      {
        v6 = 0;
        v9 = 0;
      }
      else
      {
        v5 = sub_BD2BC0(a1);
        v7 = v5 + v6;
        if ( *(char *)(a1 + 7) >= 0 )
        {
          v9 = v7;
          LOBYTE(v6) = v7 != 0;
        }
        else
        {
          v8 = v7 - sub_BD2BC0(a1);
          v9 = v8;
          LOBYTE(v6) = v8 != 0;
        }
      }
LABEL_7:
      v10 = v4 | (v6 << 28);
      v11 = sub_BD2CC0(96, v4 & 0x7FFFFFF | (unsigned __int64)(v9 << 32));
      if ( !v11 )
        return v11;
      goto LABEL_11;
    }
  }
  v10 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v11 = sub_BD2C40(96, v10);
  if ( v11 )
LABEL_11:
    sub_B4B340(v11, a1, v10);
  return v11;
}

// Function: sub_B59DB0
// Address: 0xb59db0
//
__int64 __fastcall sub_B59DB0(unsigned __int8 *a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  _BYTE *v12; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx

  v2 = *a1;
  if ( v2 == 40 )
  {
    v3 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v3 = 0;
    if ( v2 != 85 )
    {
      v3 = 64;
      if ( v2 != 34 )
LABEL_19:
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_10;
  v4 = sub_BD2BC0(a1);
  v6 = v4 + v5;
  if ( (a1[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v6 >> 4) )
      goto LABEL_19;
    goto LABEL_10;
  }
  if ( !(unsigned int)((v6 - sub_BD2BC0(a1)) >> 4) )
  {
LABEL_10:
    v10 = 0;
    goto LABEL_11;
  }
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_19;
  v7 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
  if ( (a1[7] & 0x80u) == 0 )
    BUG();
  v8 = sub_BD2BC0(a1);
  v10 = 32LL * (unsigned int)(*(_DWORD *)(v8 + v9 - 4) - v7);
LABEL_11:
  v11 = *(_QWORD *)&a1[32
                     * ((unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v3 - v10) >> 5)
                      - 2
                      - (unsigned __int64)(*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
  if ( *(_BYTE *)v11 != 24 )
    return 0;
  v12 = *(_BYTE **)(v11 + 24);
  if ( !v12 || *v12 )
    return 0;
  v14 = sub_B91420(v12, a2);
  return sub_E3F580(v14, v15);
}

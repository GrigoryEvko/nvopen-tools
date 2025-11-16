// Function: sub_30D3C00
// Address: 0x30d3c00
//
__int64 __fastcall sub_30D3C00(__int64 a1, unsigned __int8 *a2)
{
  int v2; // edx
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r14
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 result; // rax

  v2 = *a2;
  if ( v2 == 40 )
  {
    v3 = 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v3 = 0;
    if ( v2 != 85 )
    {
      v3 = 64;
      if ( v2 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_10;
  v4 = sub_BD2BC0((__int64)a2);
  v6 = v4 + v5;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v6 >> 4) )
LABEL_19:
      BUG();
LABEL_10:
    v10 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v6 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_19;
  v7 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v8 = sub_BD2BC0((__int64)a2);
  v10 = 32LL * (unsigned int)(*(_DWORD *)(v8 + v9 - 4) - v7);
LABEL_11:
  v11 = (unsigned int)qword_5030168 * (unsigned int)((32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF) - 32 - v3 - v10) >> 5);
  if ( v11 >= 0x80000000 )
    v11 = 0x7FFFFFFF;
  result = *(int *)(a1 + 716) + v11;
  if ( result >= 0x80000000LL )
    result = 0x7FFFFFFF;
  *(_DWORD *)(a1 + 716) = result;
  return result;
}

// Function: sub_31700B0
// Address: 0x31700b0
//
__int64 __fastcall sub_31700B0(__int64 a1, unsigned __int8 *a2)
{
  int v2; // edx
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  int v7; // r13d
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rbx
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
LABEL_24:
      BUG();
LABEL_10:
    v10 = 0;
    goto LABEL_11;
  }
  if ( !(unsigned int)((v6 - sub_BD2BC0((__int64)a2)) >> 4) )
    goto LABEL_10;
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_24;
  v7 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
  if ( (a2[7] & 0x80u) == 0 )
    BUG();
  v8 = sub_BD2BC0((__int64)a2);
  v10 = 32LL * (unsigned int)(*(_DWORD *)(v8 + v9 - 4) - v7);
LABEL_11:
  v11 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
  v12 = (32 * v11 - 32 - v3 - v10) >> 5;
  if ( (_DWORD)v12 )
  {
    v13 = 0;
    while ( 1 )
    {
      if ( *(_QWORD *)&a2[32 * (v13 - v11)] == **(_QWORD **)(a1 + 336) && sub_B49EE0(a2, v13) )
      {
        ++v13;
        *(_QWORD *)(a1 + 16) = a2;
        if ( v13 == (unsigned int)v12 )
          break;
      }
      else if ( ++v13 == (unsigned int)v12 )
      {
        break;
      }
      v11 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
    }
  }
  result = sub_B19DB0(*(_QWORD *)(a1 + 368), **(_QWORD **)(a1 + 376), (__int64)a2);
  if ( !(_BYTE)result )
    *(_BYTE *)(a1 + 696) = 1;
  return result;
}

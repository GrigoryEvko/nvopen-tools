// Function: sub_C462E0
// Address: 0xc462e0
//
unsigned __int64 __fastcall sub_C462E0(__int64 a1, signed __int64 a2)
{
  unsigned int v2; // eax
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // r12
  unsigned __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned __int64 v11; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-28h]
  unsigned __int64 v13; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-18h]

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(_QWORD *)a1;
  if ( v2 > 0x40 )
    v4 = *(_QWORD *)(v3 + 8LL * ((v2 - 1) >> 6));
  else
    v4 = *(_QWORD *)a1;
  if ( (v4 & (1LL << ((unsigned __int8)v2 - 1))) != 0 )
  {
    v12 = *(_DWORD *)(a1 + 8);
    if ( a2 >= 0 )
    {
      if ( v2 > 0x40 )
      {
        sub_C43780((__int64)&v11, (const void **)a1);
        v2 = v12;
        if ( v12 > 0x40 )
        {
          sub_C43D10((__int64)&v11);
LABEL_9:
          sub_C46250((__int64)&v11);
          v6 = v12;
          v12 = 0;
          v14 = v6;
          v13 = v11;
          goto LABEL_10;
        }
        v3 = v11;
      }
      v5 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & ~v3;
      if ( !v2 )
        v5 = 0;
      v11 = v5;
      goto LABEL_9;
    }
    if ( v2 > 0x40 )
    {
      sub_C43780((__int64)&v11, (const void **)a1);
      v2 = v12;
      if ( v12 > 0x40 )
      {
        sub_C43D10((__int64)&v11);
LABEL_21:
        sub_C46250((__int64)&v11);
        v10 = v12;
        v12 = 0;
        a2 = -a2;
        v14 = v10;
        v13 = v11;
LABEL_10:
        v7 = -(__int64)sub_C459C0((__int64)&v13, a2);
        if ( v14 > 0x40 && v13 )
          j_j___libc_free_0_0(v13);
        if ( v12 > 0x40 )
        {
          if ( v11 )
            j_j___libc_free_0_0(v11);
        }
        return v7;
      }
      v3 = v11;
    }
    v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v2) & ~v3;
    if ( !v2 )
      v9 = 0;
    v11 = v9;
    goto LABEL_21;
  }
  if ( a2 < 0 )
    a2 = -a2;
  return sub_C459C0(a1, a2);
}

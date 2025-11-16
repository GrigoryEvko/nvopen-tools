// Function: sub_C46780
// Address: 0xc46780
//
__int64 __fastcall sub_C46780(__int64 a1, signed __int64 a2, __int64 a3, unsigned __int64 *a4)
{
  unsigned __int64 v6; // rdx
  unsigned int v7; // eax
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned int v11; // edx
  unsigned __int64 v12; // rax
  __int64 result; // rax
  unsigned __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v17; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-38h]
  __int64 *v19; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-28h]

  v6 = *(_QWORD *)a1;
  v16 = *a4;
  v7 = *(_DWORD *)(a1 + 8);
  if ( v7 > 0x40 )
    v8 = *(_QWORD *)(v6 + 8LL * ((v7 - 1) >> 6));
  else
    v8 = v6;
  if ( (v8 & (1LL << ((unsigned __int8)v7 - 1))) != 0 )
  {
    if ( a2 >= 0 )
    {
      v18 = *(_DWORD *)(a1 + 8);
      if ( v7 > 0x40 )
      {
        sub_C43780((__int64)&v17, (const void **)a1);
        v7 = v18;
        if ( v18 > 0x40 )
        {
          sub_C43D10((__int64)&v17);
LABEL_9:
          sub_C46250((__int64)&v17);
          v10 = v18;
          v18 = 0;
          v20 = v10;
          v19 = v17;
          sub_C45A90(&v19, a2, (unsigned __int64 *)a3, (unsigned __int64 *)&v16);
          if ( v20 > 0x40 && v19 )
            j_j___libc_free_0_0(v19);
          if ( v18 > 0x40 && v17 )
            j_j___libc_free_0_0(v17);
          v11 = *(_DWORD *)(a3 + 8);
          if ( v11 > 0x40 )
          {
            sub_C43D10(a3);
          }
          else
          {
            v12 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & ~*(_QWORD *)a3;
            if ( !v11 )
              v12 = 0;
            *(_QWORD *)a3 = v12;
          }
          sub_C46250(a3);
LABEL_20:
          result = -v16;
          *a4 = -v16;
          return result;
        }
        v6 = (unsigned __int64)v17;
      }
      v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & ~v6;
      if ( !v7 )
        v9 = 0;
      v17 = (__int64 *)v9;
      goto LABEL_9;
    }
    v18 = *(_DWORD *)(a1 + 8);
    if ( v7 > 0x40 )
    {
      sub_C43780((__int64)&v17, (const void **)a1);
      v7 = v18;
      if ( v18 > 0x40 )
      {
        sub_C43D10((__int64)&v17);
LABEL_25:
        sub_C46250((__int64)&v17);
        v15 = v18;
        v18 = 0;
        v20 = v15;
        v19 = v17;
        sub_C45A90(&v19, -a2, (unsigned __int64 *)a3, (unsigned __int64 *)&v16);
        if ( v20 > 0x40 && v19 )
          j_j___libc_free_0_0(v19);
        if ( v18 > 0x40 && v17 )
          j_j___libc_free_0_0(v17);
        goto LABEL_20;
      }
      v6 = (unsigned __int64)v17;
    }
    v14 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v7) & ~v6;
    if ( !v7 )
      v14 = 0;
    v17 = (__int64 *)v14;
    goto LABEL_25;
  }
  if ( a2 < 0 )
  {
    sub_C45A90((__int64 **)a1, -a2, (unsigned __int64 *)a3, (unsigned __int64 *)&v16);
    if ( *(_DWORD *)(a3 + 8) <= 0x40u )
    {
      *(_QWORD *)a3 = ~*(_QWORD *)a3;
      sub_C43640((unsigned __int64 *)a3);
    }
    else
    {
      sub_C43D10(a3);
    }
    sub_C46250(a3);
    result = v16;
    *a4 = v16;
  }
  else
  {
    sub_C45A90((__int64 **)a1, a2, (unsigned __int64 *)a3, (unsigned __int64 *)&v16);
    result = v16;
    *a4 = v16;
  }
  return result;
}

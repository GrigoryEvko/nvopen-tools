// Function: sub_C4BD10
// Address: 0xc4bd10
//
__int64 __fastcall sub_C4BD10(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  unsigned __int64 v5; // rsi
  __int64 v6; // rdx
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rcx
  unsigned int v9; // ebx
  unsigned int v10; // edx
  unsigned __int64 v11; // rsi
  __int64 v12; // rax
  unsigned int v14; // eax
  unsigned int v15; // eax
  bool v16; // cc
  __int64 v18; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-58h]
  const void *v20; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-48h]
  const void *v22; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v5 = *(_QWORD *)a2;
  v6 = 1LL << ((unsigned __int8)v4 - 1);
  if ( v4 <= 0x40 )
  {
    v7 = v5;
    if ( (v6 & v5) == 0 )
    {
      v21 = v4;
      v20 = (const void *)v5;
      goto LABEL_8;
    }
    v23 = v4;
    goto LABEL_4;
  }
  if ( (*(_QWORD *)(v5 + 8LL * ((v4 - 1) >> 6)) & v6) == 0 )
  {
    v21 = v4;
    sub_C43780((__int64)&v20, (const void **)a2);
    goto LABEL_8;
  }
  v23 = v4;
  sub_C43780((__int64)&v22, (const void **)a2);
  v4 = v23;
  if ( v23 <= 0x40 )
  {
    v7 = (unsigned __int64)v22;
LABEL_4:
    v8 = ~v7 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v4);
    if ( !v4 )
      v8 = 0;
    v22 = (const void *)v8;
    goto LABEL_7;
  }
  sub_C43D10((__int64)&v22);
LABEL_7:
  sub_C46250((__int64)&v22);
  v21 = v23;
  v20 = v22;
LABEL_8:
  sub_C4B490((__int64)&v18, (__int64)&v20, a3);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  v9 = v19;
  if ( v19 <= 0x40 )
  {
    if ( v18 )
    {
LABEL_13:
      v10 = *(_DWORD *)(a2 + 8);
      v11 = *(_QWORD *)a2;
      v12 = 1LL << ((unsigned __int8)v10 - 1);
      if ( v10 > 0x40 )
      {
        if ( (*(_QWORD *)(v11 + 8LL * ((v10 - 1) >> 6)) & v12) != 0 )
        {
          v23 = *(_DWORD *)(a2 + 8);
          sub_C43780((__int64)&v22, (const void **)a2);
          goto LABEL_16;
        }
      }
      else if ( (v12 & v11) != 0 )
      {
        v23 = *(_DWORD *)(a2 + 8);
        v22 = (const void *)v11;
LABEL_16:
        sub_C45EE0((__int64)&v22, &v18);
        *(_DWORD *)(a1 + 8) = v23;
        *(_QWORD *)a1 = v22;
LABEL_17:
        v9 = v19;
        goto LABEL_18;
      }
      v21 = *(_DWORD *)(a3 + 8);
      if ( v21 > 0x40 )
        sub_C43780((__int64)&v20, (const void **)a3);
      else
        v20 = *(const void **)a3;
      sub_C46B40((__int64)&v20, &v18);
      v15 = v21;
      v21 = 0;
      v23 = v15;
      v22 = v20;
      sub_C45EE0((__int64)&v22, (__int64 *)a2);
      v16 = v21 <= 0x40;
      *(_DWORD *)(a1 + 8) = v23;
      *(_QWORD *)a1 = v22;
      if ( !v16 && v20 )
        j_j___libc_free_0_0(v20);
      goto LABEL_17;
    }
  }
  else if ( v9 != (unsigned int)sub_C444A0((__int64)&v18) )
  {
    goto LABEL_13;
  }
  v14 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v14;
  if ( v14 > 0x40 )
  {
    sub_C43780(a1, (const void **)a2);
    v9 = v19;
  }
  else
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
  }
LABEL_18:
  if ( v9 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return a1;
}

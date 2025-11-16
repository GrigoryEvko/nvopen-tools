// Function: sub_2ECB630
// Address: 0x2ecb630
//
unsigned __int64 *__fastcall sub_2ECB630(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  const char *v4; // rbx
  __int64 v5; // rcx
  __int64 v6; // rax
  unsigned __int64 *v7; // r12
  const char *v8; // rdi
  char *v9; // rsi
  __int64 (__fastcall *v10)(__int64); // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rdi
  char *v14; // rsi
  __int64 v15; // [rsp+8h] [rbp-58h] BYREF
  const char *v16; // [rsp+10h] [rbp-50h] BYREF
  const char *v17; // [rsp+18h] [rbp-48h]
  char v18; // [rsp+30h] [rbp-30h]
  char v19; // [rsp+31h] [rbp-2Fh]

  v2 = sub_22077B0(0x690u);
  v4 = (const char *)v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = a1;
    *(_QWORD *)(v2 + 16) = 0;
    *(_QWORD *)(v2 + 24) = 0;
    *(_DWORD *)(v2 + 32) = 0;
    *(_WORD *)(v2 + 36) = 0;
    *(_QWORD *)(v2 + 56) = v2 + 72;
    *(_QWORD *)(v2 + 64) = 0x1000000000LL;
    *(_QWORD *)(v2 + 40) = 0;
    *(_BYTE *)(v2 + 52) = 0;
    *(_QWORD *)v2 = &unk_4A29F38;
    v16 = "TopQ";
    *(_DWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 136) = 0;
    v19 = 1;
    v18 = 3;
    sub_2EC8790(v2 + 144, 1, (char *)&v16, v3);
    v19 = 1;
    v16 = "BotQ";
    v18 = 3;
    sub_2EC8790((__int64)(v4 + 864), 2, (char *)&v16, v5);
    *((_BYTE *)v4 + 1584) = 0;
    *(_QWORD *)(v4 + 1588) = 0;
    *((_QWORD *)v4 + 200) = 0;
    *((_QWORD *)v4 + 201) = 0;
    *((_DWORD *)v4 + 404) = 0;
    *((_WORD *)v4 + 810) = 0;
    *((_QWORD *)v4 + 203) = 0;
    *((_BYTE *)v4 + 1632) = 0;
    *(_QWORD *)(v4 + 1636) = 0;
    *((_QWORD *)v4 + 209) = 0;
    *((_QWORD *)v4 + 206) = 0;
    *((_QWORD *)v4 + 207) = 0;
    *((_DWORD *)v4 + 416) = 0;
    *((_WORD *)v4 + 834) = 0;
  }
  v16 = v4;
  v6 = sub_22077B0(0x1A08u);
  v7 = (unsigned __int64 *)v6;
  if ( v6 )
    sub_2EC4D40(v6, a1, (__int64 *)&v16);
  if ( v16 )
    (*(void (__fastcall **)(const char *))(*(_QWORD *)v16 + 16LL))(v16);
  sub_2EC81F0(&v16);
  v8 = v16;
  if ( v16 )
  {
    v9 = (char *)v7[436];
    if ( v9 == (char *)v7[437] )
    {
      sub_2ECB480(v7 + 435, v9, &v16);
      v8 = v16;
      if ( !v16 )
        goto LABEL_11;
    }
    else
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = v16;
        v7[436] += 8LL;
        goto LABEL_11;
      }
      v7[436] = 8;
    }
    (*(void (__fastcall **)(const char *))(*(_QWORD *)v8 + 16LL))(v8);
  }
LABEL_11:
  v10 = *(__int64 (__fastcall **)(__int64))(**(_QWORD **)(a1[1] + 16LL) + 480LL);
  if ( v10 == sub_2EC09A0 )
    return v7;
  v10((__int64)&v16);
  v12 = (unsigned __int64)v16;
  if ( v17 != v16 )
  {
    sub_2F06BD0(&v15, v16, (v17 - v16) >> 3, 0);
    v13 = v15;
    if ( !v15 )
    {
LABEL_20:
      v12 = (unsigned __int64)v16;
      goto LABEL_21;
    }
    v14 = (char *)v7[436];
    if ( v14 == (char *)v7[437] )
    {
      sub_2ECB480(v7 + 435, v14, &v15);
      v13 = v15;
      if ( !v15 )
        goto LABEL_20;
    }
    else
    {
      if ( v14 )
      {
        *(_QWORD *)v14 = v15;
        v7[436] += 8LL;
        goto LABEL_20;
      }
      v7[436] = 8;
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL))(v13);
    v12 = (unsigned __int64)v16;
  }
LABEL_21:
  if ( !v12 )
    return v7;
  j_j___libc_free_0(v12);
  return v7;
}

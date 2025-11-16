// Function: sub_D5DD50
// Address: 0xd5dd50
//
__int64 __fastcall sub_D5DD50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r13
  unsigned __int8 v7; // al
  char v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  const void *v11; // rsi
  unsigned __int16 v12; // r15
  unsigned int v13; // eax
  unsigned __int64 v14; // rdx
  unsigned int v15; // eax
  unsigned int v17; // eax
  __int64 v18; // [rsp+8h] [rbp-78h]
  const void *v19; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-68h]
  const void *v21; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-58h]
  const void *v23; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-48h]
  unsigned __int64 v25; // [rsp+40h] [rbp-40h] BYREF
  __int64 v26; // [rsp+48h] [rbp-38h]

  v5 = sub_B2BCD0(a3);
  if ( !v5
    || (v6 = v5, v7 = *(_BYTE *)(v5 + 8), v7 != 12)
    && v7 > 3u
    && v7 != 5
    && (v7 & 0xFD) != 4
    && (v7 & 0xFB) != 0xA
    && ((unsigned __int8)(v7 - 15) > 3u && v7 != 20 || !(unsigned __int8)sub_BCEBA0(v6, 0)) )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  v18 = *(_QWORD *)a2;
  v8 = sub_AE5020(*(_QWORD *)a2, v6);
  v9 = sub_9208B0(v18, v6);
  v26 = v10;
  v25 = ((1LL << v8) + ((unsigned __int64)(v9 + 7) >> 3) - 1) >> v8 << v8;
  v11 = (const void *)sub_CA1930(&v25);
  v20 = *(_DWORD *)(a2 + 32);
  if ( v20 > 0x40 )
    sub_C43690((__int64)&v19, (__int64)v11, 0);
  else
    v19 = v11;
  v12 = sub_B2BD00(a3);
  v22 = v20;
  if ( v20 > 0x40 )
    sub_C43780((__int64)&v21, &v19);
  else
    v21 = v19;
  sub_D5D630((__int64)&v23, a2, &v21, v12);
  v13 = *(_DWORD *)(a2 + 48);
  LODWORD(v26) = v13;
  if ( v13 > 0x40 )
  {
    sub_C43780((__int64)&v25, (const void **)(a2 + 40));
    v17 = v26;
    *(_DWORD *)(a1 + 8) = v26;
    if ( v17 > 0x40 )
    {
      sub_C43780(a1, (const void **)&v25);
      goto LABEL_11;
    }
  }
  else
  {
    v14 = *(_QWORD *)(a2 + 40);
    *(_DWORD *)(a1 + 8) = v13;
    v25 = v14;
  }
  *(_QWORD *)a1 = v25;
LABEL_11:
  v15 = v24;
  *(_DWORD *)(a1 + 24) = v24;
  if ( v15 > 0x40 )
    sub_C43780(a1 + 16, &v23);
  else
    *(_QWORD *)(a1 + 16) = v23;
  if ( (unsigned int)v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return a1;
}

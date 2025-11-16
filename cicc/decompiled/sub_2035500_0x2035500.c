// Function: sub_2035500
// Address: 0x2035500
//
__int64 __fastcall sub_2035500(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  char *v6; // rax
  char v7; // r15
  const void **v8; // r8
  unsigned __int64 *v9; // rax
  unsigned __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 result; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rdx
  bool v18; // al
  __int64 v19; // rsi
  __int64 *v20; // rbx
  __int64 v21; // rsi
  __int128 v22; // [rsp-10h] [rbp-70h]
  __int128 v23; // [rsp-10h] [rbp-70h]
  const void **v24; // [rsp+0h] [rbp-60h]
  __int64 v25; // [rsp+8h] [rbp-58h]
  unsigned int v26; // [rsp+10h] [rbp-50h] BYREF
  const void **v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+20h] [rbp-40h] BYREF
  int v29; // [rsp+28h] [rbp-38h]

  v6 = *(char **)(a2 + 40);
  v7 = *v6;
  v8 = (const void **)*((_QWORD *)v6 + 1);
  v9 = *(unsigned __int64 **)(a2 + 32);
  LOBYTE(v26) = v7;
  v10 = *v9;
  v11 = v9[1];
  v27 = v8;
  v24 = v8;
  result = sub_2032580(a1, v10, v11);
  v13 = a1;
  v15 = v14;
  v16 = result;
  v17 = *(_QWORD *)(result + 40) + 16LL * (unsigned int)v14;
  if ( *(_BYTE *)v17 == v7 )
  {
    if ( *(const void ***)(v17 + 8) == v24 || v7 )
      return result;
  }
  else if ( v7 )
  {
    v18 = (unsigned __int8)(v7 - 86) <= 0x17u || (unsigned __int8)(v7 - 8) <= 5u;
    goto LABEL_7;
  }
  v18 = sub_1F58CD0((__int64)&v26);
  v13 = a1;
LABEL_7:
  v19 = *(_QWORD *)(a2 + 72);
  v20 = *(__int64 **)(v13 + 8);
  v28 = v19;
  if ( v18 )
  {
    if ( v19 )
      sub_1623A60((__int64)&v28, v19, 2);
    *((_QWORD *)&v22 + 1) = v15;
    *(_QWORD *)&v22 = v16;
    v29 = *(_DWORD *)(a2 + 64);
    result = sub_1D309E0(v20, 157, (__int64)&v28, v26, v27, 0, a3, a4, a5, v22);
    v21 = v28;
    if ( v28 )
    {
LABEL_11:
      v25 = result;
      sub_161E7C0((__int64)&v28, v21);
      return v25;
    }
  }
  else
  {
    if ( v19 )
      sub_1623A60((__int64)&v28, v19, 2);
    *((_QWORD *)&v23 + 1) = v15;
    *(_QWORD *)&v23 = v16;
    v29 = *(_DWORD *)(a2 + 64);
    result = sub_1D309E0(v20, 144, (__int64)&v28, v26, v27, 0, a3, a4, a5, v23);
    v21 = v28;
    if ( v28 )
      goto LABEL_11;
  }
  return result;
}

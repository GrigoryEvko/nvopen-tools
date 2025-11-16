// Function: sub_22C7770
// Address: 0x22c7770
//
__int64 __fastcall sub_22C7770(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v7; // r14
  unsigned int v8; // eax
  unsigned int v9; // esi
  unsigned int v10; // eax
  const void *v11; // rcx
  unsigned int v12; // edx
  bool v13; // zf
  __int64 v14; // rdx
  __int64 *v15; // rax
  unsigned int v16; // [rsp+Ch] [rbp-74h]
  __int64 v17; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-68h]
  const void *v19; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-58h]
  unsigned __int8 v21[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v22; // [rsp+38h] [rbp-48h] BYREF
  unsigned int v23; // [rsp+40h] [rbp-40h]
  const void *v24; // [rsp+48h] [rbp-38h] BYREF
  unsigned int v25; // [rsp+50h] [rbp-30h]
  char v26; // [rsp+58h] [rbp-28h]

  sub_22C7100((__int64)v21, a2, a3, a5, a4);
  if ( !v26 )
  {
    *(_BYTE *)(a1 + 32) = 0;
    return a1;
  }
  v7 = v21[0];
  if ( v21[0] == 4
    || (v8 = sub_BCB060(*(_QWORD *)(a3 + 8)), v9 = v8, v7 == 5)
    && (v16 = v8, v15 = sub_9876C0(&v22), v7 = v21[0], v9 = v16, v15) )
  {
    v18 = v23;
    if ( v23 > 0x40 )
      sub_C43780((__int64)&v17, (const void **)&v22);
    else
      v17 = v22;
    v10 = v25;
    v20 = v25;
    if ( v25 > 0x40 )
    {
      sub_C43780((__int64)&v19, &v24);
      v10 = v20;
      v11 = v19;
    }
    else
    {
      v11 = v24;
    }
  }
  else if ( v7 == 2 )
  {
    sub_AD8380((__int64)&v17, v22);
    v10 = v20;
    v11 = v19;
  }
  else
  {
    if ( v7 )
      sub_AADB10((__int64)&v17, v9, 1);
    else
      sub_AADB10((__int64)&v17, v9, 0);
    v10 = v20;
    v11 = v19;
  }
  v12 = v18;
  v13 = v26 == 0;
  *(_DWORD *)(a1 + 24) = v10;
  *(_QWORD *)(a1 + 16) = v11;
  *(_DWORD *)(a1 + 8) = v12;
  v14 = v17;
  *(_BYTE *)(a1 + 32) = 1;
  *(_QWORD *)a1 = v14;
  if ( v13 )
    return a1;
  v26 = 0;
  sub_22C0090(v21);
  return a1;
}

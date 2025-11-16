// Function: sub_19D6B50
// Address: 0x19d6b50
//
__int64 __fastcall sub_19D6B50(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdx
  int v10; // eax
  __int64 v11; // r13
  unsigned int v12; // r14d
  __int64 v13; // r15
  int v14; // eax
  int v15; // eax
  unsigned int v16; // eax
  __int64 v17; // r12
  int v18; // eax
  __int64 v19; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+28h] [rbp-48h]
  __int64 v25; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 32);
  v3 = *(_QWORD *)(a2 + 32);
  if ( v2 && (v4 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF))) != 0 )
  {
    if ( !v3 )
    {
LABEL_5:
      LODWORD(v5) = 0;
      return (unsigned int)v5;
    }
  }
  else
  {
    v4 = 0;
    if ( !v3 )
      goto LABEL_8;
  }
  if ( *(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)) != v4 )
    goto LABEL_5;
LABEL_8:
  v7 = *(_QWORD *)(a1 + 64);
  v8 = *(_QWORD *)(a2 + 64);
  if ( v7 && (v9 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF))) != 0 )
  {
    if ( !v8 )
      goto LABEL_5;
  }
  else
  {
    v9 = 0;
    if ( !v8 )
      goto LABEL_12;
  }
  if ( *(_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF)) != v9 )
    goto LABEL_5;
LABEL_12:
  v10 = *(_DWORD *)(a1 + 96);
  v20 = *(_DWORD *)(a1 + 56);
  v11 = v10 / 8;
  if ( v20 > 0x40 )
    sub_16A4FD0((__int64)&v19, (const void **)(a1 + 48));
  else
    v19 = *(_QWORD *)(a1 + 48);
  sub_16A7490((__int64)&v19, v11);
  v12 = v20;
  v13 = v19;
  v20 = 0;
  v22 = v12;
  v21 = v19;
  if ( v12 <= 0x40 )
  {
    if ( v19 != *(_QWORD *)(a2 + 48) )
      goto LABEL_5;
LABEL_25:
    v15 = *(_DWORD *)(a1 + 96);
    v24 = *(_DWORD *)(a1 + 88);
    v5 = v15 / 8;
    if ( v24 > 0x40 )
      sub_16A4FD0((__int64)&v23, (const void **)(a1 + 80));
    else
      v23 = *(_QWORD *)(a1 + 80);
    sub_16A7490((__int64)&v23, v5);
    v16 = v24;
    v17 = v23;
    v24 = 0;
    v26 = v16;
    v25 = v23;
    if ( v16 <= 0x40 )
    {
      LOBYTE(v5) = v23 == *(_QWORD *)(a2 + 80);
    }
    else
    {
      LOBYTE(v18) = sub_16A5220((__int64)&v25, (const void **)(a2 + 80));
      LODWORD(v5) = v18;
      if ( v17 )
      {
        j_j___libc_free_0_0(v17);
        if ( v24 > 0x40 )
        {
          if ( v23 )
            j_j___libc_free_0_0(v23);
        }
      }
    }
    if ( v12 <= 0x40 )
      goto LABEL_18;
    goto LABEL_16;
  }
  LOBYTE(v14) = sub_16A5220((__int64)&v21, (const void **)(a2 + 48));
  LODWORD(v5) = v14;
  if ( (_BYTE)v14 )
    goto LABEL_25;
LABEL_16:
  if ( v13 )
    j_j___libc_free_0_0(v13);
LABEL_18:
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return (unsigned int)v5;
}

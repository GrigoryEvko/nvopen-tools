// Function: sub_1D663F0
// Address: 0x1d663f0
//
__int64 __fastcall sub_1D663F0(__int64 a1, __int64 a2)
{
  __int64 v4; // r13
  unsigned int v5; // r15d
  __int64 *v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdi
  unsigned __int8 v15; // al
  __int64 v16; // rsi
  _BYTE *v17; // rsi
  _BYTE *v18; // rcx
  __int64 v19; // rax
  unsigned int v20; // eax
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 *v23; // [rsp+8h] [rbp-98h]
  _BYTE *v24; // [rsp+10h] [rbp-90h]
  __int64 v25; // [rsp+10h] [rbp-90h]
  __int64 v26; // [rsp+18h] [rbp-88h]
  _BYTE *v27; // [rsp+18h] [rbp-88h]
  _BYTE *v28; // [rsp+18h] [rbp-88h]
  __int64 *v29; // [rsp+18h] [rbp-88h]
  __int64 *v30; // [rsp+20h] [rbp-80h]
  __int64 v31; // [rsp+28h] [rbp-78h]
  __int64 v32; // [rsp+28h] [rbp-78h]
  __int64 v33; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v34; // [rsp+38h] [rbp-68h]
  __int64 v35; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v36; // [rsp+48h] [rbp-58h]
  _BYTE *v37; // [rsp+50h] [rbp-50h] BYREF
  _BYTE *v38; // [rsp+58h] [rbp-48h]
  _BYTE *v39; // [rsp+60h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 40);
  if ( *(_BYTE *)(sub_157EBA0(v4) + 16) != 28 )
    return 0;
  v5 = sub_1D5B320(a1);
  if ( !(_BYTE)v5 )
    return 0;
  v7 = *(__int64 **)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
  v30 = v7 + 3;
  if ( (int)sub_14A30B0(a2) > 1 )
    return 0;
  v8 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  v26 = v8;
  if ( *(_BYTE *)(v8 + 16) <= 0x17u )
    return 0;
  if ( v4 != *(_QWORD *)(v8 + 40) )
    return 0;
  v9 = *(_QWORD *)(a1 + 8);
  if ( !v9 )
    return 0;
  while ( 1 )
  {
    v31 = v9;
    v10 = sub_1648700(v9);
    if ( *((_BYTE *)v10 + 16) > 0x17u && v4 != v10[5] )
      break;
    v9 = *(_QWORD *)(v31 + 8);
    if ( !v9 )
      return 0;
  }
  v38 = 0;
  v39 = 0;
  v11 = *(_QWORD *)(v26 + 8);
  v37 = 0;
  if ( v11 )
  {
    do
    {
      v32 = v11;
      v12 = sub_1648700(v11);
      v13 = v32;
      v14 = (__int64)v12;
      if ( (_QWORD *)a1 != v12 )
      {
        v15 = *((_BYTE *)v12 + 16);
        if ( v15 <= 0x17u )
          goto LABEL_16;
        if ( *(_QWORD *)(v14 + 40) != v4 )
        {
          if ( v15 != 56 )
            goto LABEL_16;
          v35 = v14;
          if ( !(unsigned __int8)sub_1D5B320(v14) )
            goto LABEL_16;
          v16 = *(_QWORD *)(v35 - 24LL * (*(_DWORD *)(v35 + 20) & 0xFFFFFFF));
          if ( v26 != v16
            || !v16
            || **(_QWORD **)(v35 + 24 * (1LL - (*(_DWORD *)(v35 + 20) & 0xFFFFFFF))) != *v7
            || (int)sub_14A30B0(a2) > 1 )
          {
            goto LABEL_16;
          }
          v17 = v38;
          v13 = v32;
          if ( v38 == v39 )
          {
            sub_1CADA60((__int64)&v37, v38, &v35);
            v13 = v32;
          }
          else
          {
            if ( v38 )
            {
              *(_QWORD *)v38 = v35;
              v17 = v38;
            }
            v38 = v17 + 8;
          }
        }
      }
      v11 = *(_QWORD *)(v13 + 8);
    }
    while ( v11 );
    v18 = v37;
    v24 = v38;
    if ( v38 == v37 )
    {
LABEL_16:
      v5 = 0;
      goto LABEL_17;
    }
    do
    {
      v19 = *(_QWORD *)(*(_QWORD *)v18 + 24 * (1LL - (*(_DWORD *)(*(_QWORD *)v18 + 20LL) & 0xFFFFFFF)));
      v36 = *(_DWORD *)(v19 + 32);
      if ( v36 <= 0x40 )
      {
        v35 = *(_QWORD *)(v19 + 24);
      }
      else
      {
        v28 = v18;
        sub_16A4FD0((__int64)&v35, (const void **)(v19 + 24));
        v18 = v28;
      }
      v27 = v18;
      sub_16A7590((__int64)&v35, v30);
      v34 = v36;
      v33 = v35;
      if ( (unsigned int)sub_14A30B0(a2) > 1 )
      {
        v5 = 0;
        sub_135E100(&v33);
        goto LABEL_17;
      }
      sub_135E100(&v33);
      v18 = v27 + 8;
    }
    while ( v24 != v27 + 8 );
    v23 = (__int64 *)v38;
    v29 = (__int64 *)v37;
    if ( v38 != v37 )
    {
      do
      {
        v21 = *v29;
        sub_1593B40((_QWORD *)(*v29 - 24LL * (*(_DWORD *)(*v29 + 20) & 0xFFFFFFF)), a1);
        v22 = *(_QWORD *)(v21 + 24 * (1LL - (*(_DWORD *)(v21 + 20) & 0xFFFFFFF)));
        v34 = *(_DWORD *)(v22 + 32);
        if ( v34 <= 0x40 )
          v33 = *(_QWORD *)(v22 + 24);
        else
          sub_16A4FD0((__int64)&v33, (const void **)(v22 + 24));
        sub_16A7590((__int64)&v33, v30);
        v20 = v34;
        v34 = 0;
        v36 = v20;
        v35 = v33;
        v25 = sub_15A1070(*v7, (__int64)&v35);
        sub_135E100(&v35);
        sub_135E100(&v33);
        sub_1593B40((_QWORD *)(v21 + 24 * (1LL - (*(_DWORD *)(v21 + 20) & 0xFFFFFFF))), v25);
        if ( !sub_15FA300(a1) )
          sub_15FA2E0(v21, 0);
        ++v29;
      }
      while ( v23 != v29 );
    }
LABEL_17:
    if ( v37 )
      j_j___libc_free_0(v37, v39 - v37);
  }
  else
  {
    return 0;
  }
  return v5;
}

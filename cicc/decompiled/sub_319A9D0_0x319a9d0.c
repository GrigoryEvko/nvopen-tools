// Function: sub_319A9D0
// Address: 0x319a9d0
//
unsigned __int64 __fastcall sub_319A9D0(__int64 *a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // r13
  __int64 **v9; // r15
  __int64 v10; // rdi
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v12; // r14
  __int64 **v13; // r15
  __int64 v14; // rdi
  __int64 (__fastcall *v15)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v16; // r14
  __int64 v17; // rdx
  int v18; // r13d
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // r14
  __int64 v24; // rdx
  int v25; // r12d
  __int64 v26; // r12
  __int64 v27; // rbx
  __int64 v28; // rdx
  unsigned int v29; // esi
  _BYTE v30[32]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v31; // [rsp+30h] [rbp-70h]
  _BYTE v32[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v33; // [rsp+60h] [rbp-40h]

  v4 = a2;
  if ( *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8 == *(_DWORD *)(*(_QWORD *)(a3 + 8) + 8LL) >> 8 )
    return v4;
  if ( !(unsigned __int8)sub_DF97F0(a4) )
  {
    if ( !(unsigned __int8)sub_DF97F0(a4) )
      BUG();
    v13 = *(__int64 ***)(a3 + 8);
    v31 = 257;
    if ( v13 == *(__int64 ***)(a2 + 8) )
      return a2;
    v14 = a1[10];
    v15 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v14 + 120LL);
    if ( v15 == sub_920130 )
    {
      if ( *(_BYTE *)a2 > 0x15u )
        goto LABEL_19;
      if ( (unsigned __int8)sub_AC4810(0x32u) )
        v16 = sub_ADAB70(50, a2, v13, 0);
      else
        v16 = sub_AA93C0(0x32u, a2, (__int64)v13);
    }
    else
    {
      v16 = v15(v14, 50u, (_BYTE *)a2, (__int64)v13);
    }
    if ( v16 )
      return v16;
LABEL_19:
    v33 = 257;
    v16 = sub_B51D30(50, a2, (__int64)v13, (__int64)v32, 0, 0);
    if ( (unsigned __int8)sub_920620(v16) )
    {
      v17 = a1[12];
      v18 = *((_DWORD *)a1 + 26);
      if ( v17 )
        sub_B99FD0(v16, 3u, v17);
      sub_B45150(v16, v18);
    }
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v16,
      v30,
      a1[7],
      a1[8]);
    v19 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v19 )
    {
      v20 = *a1;
      do
      {
        v21 = *(_QWORD *)(v20 + 8);
        v22 = *(_DWORD *)v20;
        v20 += 16;
        sub_B99FD0(v16, v22, v21);
      }
      while ( v19 != v20 );
    }
    return v16;
  }
  v9 = *(__int64 ***)(a2 + 8);
  v31 = 257;
  if ( v9 == *(__int64 ***)(a3 + 8) )
    return v4;
  v10 = a1[10];
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v10 + 120LL);
  if ( v11 != sub_920130 )
  {
    v12 = v11(v10, 50u, (_BYTE *)a3, (__int64)v9);
    goto LABEL_9;
  }
  if ( *(_BYTE *)a3 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x32u) )
      v12 = sub_ADAB70(50, a3, v9, 0);
    else
      v12 = sub_AA93C0(0x32u, a3, (__int64)v9);
LABEL_9:
    if ( v12 )
      return v4;
  }
  v33 = 257;
  v23 = sub_B51D30(50, a3, (__int64)v9, (__int64)v32, 0, 0);
  if ( (unsigned __int8)sub_920620(v23) )
  {
    v24 = a1[12];
    v25 = *((_DWORD *)a1 + 26);
    if ( v24 )
      sub_B99FD0(v23, 3u, v24);
    sub_B45150(v23, v25);
  }
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v23,
    v30,
    a1[7],
    a1[8]);
  v26 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v26 )
  {
    v27 = *a1;
    do
    {
      v28 = *(_QWORD *)(v27 + 8);
      v29 = *(_DWORD *)v27;
      v27 += 16;
      sub_B99FD0(v23, v29, v28);
    }
    while ( v26 != v27 );
  }
  return v4;
}

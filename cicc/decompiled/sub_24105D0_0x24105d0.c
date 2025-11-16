// Function: sub_24105D0
// Address: 0x24105d0
//
__int64 __fastcall sub_24105D0(__int64 *a1, unsigned int a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, __int64, __int64); // rax
  unsigned int *v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rdx
  unsigned int v13; // esi
  __int64 **v14; // r14
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v17; // r13
  __int64 v19; // rdx
  int v20; // r14d
  unsigned int *v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rdx
  unsigned int v24; // esi
  _BYTE *v25; // rax
  unsigned int *v26; // r13
  __int64 v27; // rbx
  __int64 v28; // rdx
  unsigned int v29; // esi
  _QWORD v31[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v32; // [rsp+30h] [rbp-70h]
  _BYTE v33[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v5 = *a1;
  v32 = 257;
  v6 = *(_QWORD *)(v5 + 80);
  v7 = *(_QWORD *)(v5 + 64);
  if ( v7 != *(_QWORD *)(v6 + 8) )
  {
    if ( *(_BYTE *)v6 <= 0x15u )
    {
      v8 = *(_QWORD *)(a3 + 80);
      v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v8 + 136LL);
      if ( v9 == sub_928970 )
        v6 = sub_ADAFB0(v6, v7);
      else
        v6 = v9(v8, v6, v7);
      if ( *(_BYTE *)v6 > 0x1Cu )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
          *(_QWORD *)(a3 + 88),
          v6,
          v31,
          *(_QWORD *)(a3 + 56),
          *(_QWORD *)(a3 + 64));
        v10 = *(unsigned int **)a3;
        v11 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
        if ( *(_QWORD *)a3 != v11 )
        {
          do
          {
            v12 = *((_QWORD *)v10 + 1);
            v13 = *v10;
            v10 += 4;
            sub_B99FD0(v6, v13, v12);
          }
          while ( (unsigned int *)v11 != v10 );
        }
      }
      goto LABEL_8;
    }
    v34 = 257;
    v6 = sub_B52210(v6, v7, (__int64)v33, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
      *(_QWORD *)(a3 + 88),
      v6,
      v31,
      *(_QWORD *)(a3 + 56),
      *(_QWORD *)(a3 + 64));
    v26 = *(unsigned int **)a3;
    v27 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
    if ( *(_QWORD *)a3 == v27 )
    {
LABEL_8:
      v5 = *a1;
      goto LABEL_9;
    }
    do
    {
      v28 = *((_QWORD *)v26 + 1);
      v29 = *v26;
      v26 += 4;
      sub_B99FD0(v6, v29, v28);
    }
    while ( (unsigned int *)v27 != v26 );
    v5 = *a1;
  }
LABEL_9:
  if ( a2 )
  {
    v34 = 257;
    v25 = (_BYTE *)sub_ACD640(*(_QWORD *)(v5 + 64), a2, 0);
    v6 = sub_929C50((unsigned int **)a3, (_BYTE *)v6, v25, (__int64)v33, 0, 0);
    v5 = *a1;
  }
  v31[0] = "_dfsarg";
  v32 = 259;
  v14 = (__int64 **)sub_BCE3C0(*(__int64 **)(v5 + 8), 0);
  if ( v14 == *(__int64 ***)(v6 + 8) )
    return v6;
  v15 = *(_QWORD *)(a3 + 80);
  v16 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v15 + 120LL);
  if ( v16 != sub_920130 )
  {
    v17 = v16(v15, 48u, (_BYTE *)v6, (__int64)v14);
    goto LABEL_16;
  }
  if ( *(_BYTE *)v6 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x30u) )
      v17 = sub_ADAB70(48, v6, v14, 0);
    else
      v17 = sub_AA93C0(0x30u, v6, (__int64)v14);
LABEL_16:
    if ( v17 )
      return v17;
  }
  v34 = 257;
  v17 = sub_B51D30(48, v6, (__int64)v14, (__int64)v33, 0, 0);
  if ( (unsigned __int8)sub_920620(v17) )
  {
    v19 = *(_QWORD *)(a3 + 96);
    v20 = *(_DWORD *)(a3 + 104);
    if ( v19 )
      sub_B99FD0(v17, 3u, v19);
    sub_B45150(v17, v20);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v17,
    v31,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v21 = *(unsigned int **)a3;
  v22 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  while ( (unsigned int *)v22 != v21 )
  {
    v23 = *((_QWORD *)v21 + 1);
    v24 = *v21;
    v21 += 4;
    sub_B99FD0(v17, v24, v23);
  }
  return v17;
}

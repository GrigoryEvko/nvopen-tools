// Function: sub_2784C30
// Address: 0x2784c30
//
unsigned __int64 __fastcall sub_2784C30(__int64 *a1, unsigned __int64 a2, __int64 **a3, __int64 a4)
{
  __int64 v6; // r15
  unsigned int v7; // r14d
  unsigned int v8; // eax
  __int64 v9; // r14
  __int64 v11; // rdi
  __int64 (__fastcall *v12)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v13; // rdx
  int v14; // r12d
  __int64 v15; // r12
  __int64 v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // rdi
  __int64 (__fastcall *v20)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v21; // r12
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rdx
  unsigned int v25; // esi
  _BYTE v27[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v28; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 8);
  v7 = sub_BCB060(v6);
  v8 = sub_BCB060((__int64)a3);
  if ( v7 >= v8 )
  {
    if ( v7 == v8 || (__int64 **)v6 == a3 )
      return a2;
    v19 = a1[10];
    v20 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v19 + 120LL);
    if ( v20 == sub_920130 )
    {
      if ( *(_BYTE *)a2 > 0x15u )
      {
LABEL_25:
        v28 = 257;
        v9 = sub_B51D30(38, a2, (__int64)a3, (__int64)v27, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
          a1[11],
          v9,
          a4,
          a1[7],
          a1[8]);
        v21 = 16LL * *((unsigned int *)a1 + 2);
        v22 = *a1;
        v23 = v22 + v21;
        while ( v23 != v22 )
        {
          v24 = *(_QWORD *)(v22 + 8);
          v25 = *(_DWORD *)v22;
          v22 += 16;
          sub_B99FD0(v9, v25, v24);
        }
        return v9;
      }
      if ( (unsigned __int8)sub_AC4810(0x26u) )
        v9 = sub_ADAB70(38, a2, a3, 0);
      else
        v9 = sub_AA93C0(0x26u, a2, (__int64)a3);
    }
    else
    {
      v9 = v20(v19, 38u, (_BYTE *)a2, (__int64)a3);
    }
    if ( v9 )
      return v9;
    goto LABEL_25;
  }
  if ( (__int64 **)v6 == a3 )
    return a2;
  v11 = a1[10];
  v12 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v11 + 120LL);
  if ( v12 != sub_920130 )
  {
    v9 = v12(v11, 40u, (_BYTE *)a2, (__int64)a3);
    goto LABEL_11;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x28u) )
      v9 = sub_ADAB70(40, a2, a3, 0);
    else
      v9 = sub_AA93C0(0x28u, a2, (__int64)a3);
LABEL_11:
    if ( v9 )
      return v9;
  }
  v28 = 257;
  v9 = sub_B51D30(40, a2, (__int64)a3, (__int64)v27, 0, 0);
  if ( (unsigned __int8)sub_920620(v9) )
  {
    v13 = a1[12];
    v14 = *((_DWORD *)a1 + 26);
    if ( v13 )
      sub_B99FD0(v9, 3u, v13);
    sub_B45150(v9, v14);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    a4,
    a1[7],
    a1[8]);
  v15 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v15 )
  {
    v16 = *a1;
    do
    {
      v17 = *(_QWORD *)(v16 + 8);
      v18 = *(_DWORD *)v16;
      v16 += 16;
      sub_B99FD0(v9, v18, v17);
    }
    while ( v15 != v16 );
  }
  return v9;
}

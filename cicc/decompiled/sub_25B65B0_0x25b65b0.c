// Function: sub_25B65B0
// Address: 0x25b65b0
//
unsigned __int64 __fastcall sub_25B65B0(
        __int64 a1,
        __int64 **a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        unsigned __int64 a6)
{
  __int64 *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 **v11; // r15
  __int64 (__fastcall *v12)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v13; // r12
  __int64 v15; // rdx
  int v16; // r13d
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdx
  unsigned int v20; // esi
  char v21[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v22; // [rsp+20h] [rbp-70h]
  char v23[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v24; // [rsp+50h] [rbp-40h]

  v8 = *a2;
  v22 = 257;
  v9 = sub_BCE3C0(v8, 0);
  if ( v9 == *(_QWORD *)(a6 + 8) )
    return a6;
  v10 = a4[10];
  v11 = (__int64 **)v9;
  v12 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v10 + 120LL);
  if ( v12 != sub_920130 )
  {
    v13 = v12(v10, 50u, (_BYTE *)a6, (__int64)v11);
    goto LABEL_6;
  }
  if ( *(_BYTE *)a6 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x32u) )
      v13 = sub_ADAB70(50, a6, v11, 0);
    else
      v13 = sub_AA93C0(0x32u, a6, (__int64)v11);
LABEL_6:
    if ( v13 )
      return v13;
  }
  v24 = 257;
  v13 = sub_B51D30(50, a6, (__int64)v11, (__int64)v23, 0, 0);
  if ( (unsigned __int8)sub_920620(v13) )
  {
    v15 = a4[12];
    v16 = *((_DWORD *)a4 + 26);
    if ( v15 )
      sub_B99FD0(v13, 3u, v15);
    sub_B45150(v13, v16);
  }
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a4[11] + 16LL))(
    a4[11],
    v13,
    v21,
    a4[7],
    a4[8]);
  v17 = *a4;
  v18 = *a4 + 16LL * *((unsigned int *)a4 + 2);
  if ( *a4 != v18 )
  {
    do
    {
      v19 = *(_QWORD *)(v17 + 8);
      v20 = *(_DWORD *)v17;
      v17 += 16;
      sub_B99FD0(v13, v20, v19);
    }
    while ( v18 != v17 );
  }
  return v13;
}

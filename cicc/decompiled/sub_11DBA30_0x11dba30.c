// Function: sub_11DBA30
// Address: 0x11dba30
//
unsigned __int64 __fastcall sub_11DBA30(char *a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r14
  unsigned __int64 *v5; // rcx
  unsigned __int64 v6; // r15
  __int64 v7; // r14
  unsigned int v8; // eax
  __int64 *v10; // r12
  int v11; // edx
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, unsigned int, _BYTE *, __int64); // rax
  _QWORD *v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // rdx
  unsigned int v18; // esi
  char v19; // [rsp+Fh] [rbp-A1h]
  __int64 v20; // [rsp+18h] [rbp-98h]
  int v21[8]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v22; // [rsp+40h] [rbp-70h]
  _BYTE v23[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v24; // [rsp+70h] [rbp-40h]

  v3 = 0;
  if ( (unsigned __int8)(*a1 - 72) > 1u )
    return v3;
  if ( (a1[7] & 0x40) != 0 )
    v5 = (unsigned __int64 *)*((_QWORD *)a1 - 1);
  else
    v5 = (unsigned __int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v6 = *v5;
  v19 = *a1;
  v7 = *(_QWORD *)(*v5 + 8);
  v8 = sub_BCB060(v7);
  if ( v8 >= a3 && (v19 != 73 || v8 != a3) )
    return 0;
  v10 = (__int64 *)sub_BCD140(*(_QWORD **)v7, a3);
  v11 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
  {
    BYTE4(v20) = (_BYTE)v11 == 18;
    LODWORD(v20) = *(_DWORD *)(v7 + 32);
    v10 = (__int64 *)sub_BCE1B0(v10, v20);
  }
  if ( *a1 == 73 )
  {
    v24 = 257;
    return sub_11DB4B0((__int64 *)a2, 0x28u, v6, (__int64 **)v10, (__int64)v23, 0, v21[0], 0);
  }
  v22 = 257;
  if ( v10 == *(__int64 **)(v6 + 8) )
    return v6;
  v12 = *(_QWORD *)(a2 + 80);
  v13 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v12 + 120LL);
  if ( v13 != sub_920130 )
  {
    v3 = v13(v12, 39u, (_BYTE *)v6, (__int64)v10);
    goto LABEL_18;
  }
  if ( *(_BYTE *)v6 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x27u) )
      v3 = sub_ADAB70(39, v6, (__int64 **)v10, 0);
    else
      v3 = sub_AA93C0(0x27u, v6, (__int64)v10);
LABEL_18:
    if ( v3 )
      return v3;
  }
  v24 = 257;
  v14 = sub_BD2C40(72, unk_3F10A14);
  v3 = (__int64)v14;
  if ( v14 )
    sub_B515B0((__int64)v14, v6, (__int64)v10, (__int64)v23, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v3,
    v21,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v15 = *(_QWORD *)a2;
  v16 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v16 )
  {
    do
    {
      v17 = *(_QWORD *)(v15 + 8);
      v18 = *(_DWORD *)v15;
      v15 += 16;
      sub_B99FD0(v3, v18, v17);
    }
    while ( v16 != v15 );
  }
  return v3;
}

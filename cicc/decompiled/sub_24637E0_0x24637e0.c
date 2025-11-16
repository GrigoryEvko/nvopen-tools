// Function: sub_24637E0
// Address: 0x24637e0
//
__int64 __fastcall sub_24637E0(__int64 *a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // r12
  __int64 **v5; // rax
  __int64 v6; // rbx
  int v7; // edx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int8 *v14; // r15
  __int64 (__fastcall *v15)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8); // rax
  __int64 v16; // r14
  int v17; // r12d
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 **v20; // r15
  __int64 v21; // rdi
  __int64 (__fastcall *v22)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v23; // r12
  __int64 v25; // r12
  __int64 v26; // r15
  __int64 v27; // rdx
  unsigned int v28; // esi
  int v29[8]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v30; // [rsp+30h] [rbp-70h]
  _QWORD v31[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v32; // [rsp+60h] [rbp-40h]

  v4 = a3;
  v5 = (__int64 **)sub_2463540(a1, *(_QWORD *)(a3 + 8));
  v6 = *(_QWORD *)(v4 + 8);
  if ( v5 != (__int64 **)v6 )
  {
    v7 = *(unsigned __int8 *)(v6 + 8);
    if ( (unsigned int)(v7 - 17) <= 1 )
      LOBYTE(v7) = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
    v32 = 257;
    if ( (_BYTE)v7 == 14 )
      v8 = sub_24633A0((__int64 *)a2, 0x2Fu, v4, v5, (__int64)v31, 0, v29[0], 0);
    else
      v8 = sub_24633A0((__int64 *)a2, 0x31u, v4, v5, (__int64)v31, 0, v29[0], 0);
    v6 = *(_QWORD *)(v8 + 8);
    v4 = v8;
  }
  v9 = sub_BCAE30(*(_QWORD *)(v6 + 24));
  v31[1] = v10;
  v31[0] = v9;
  v11 = sub_CA1930(v31);
  v30 = 257;
  v12 = sub_AD64C0(*(_QWORD *)(v4 + 8), (unsigned int)(v11 - 1), 0);
  v13 = *(_QWORD *)(a2 + 80);
  v14 = (unsigned __int8 *)v12;
  v15 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8))(*(_QWORD *)v13 + 24LL);
  if ( v15 != sub_920250 )
  {
    v16 = v15(v13, 27u, (_BYTE *)v4, v14, 0);
    goto LABEL_12;
  }
  if ( *(_BYTE *)v4 <= 0x15u && *v14 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(27) )
      v16 = sub_AD5570(27, v4, v14, 0, 0);
    else
      v16 = sub_AABE40(0x1Bu, (unsigned __int8 *)v4, v14);
LABEL_12:
    if ( v16 )
      goto LABEL_13;
  }
  v32 = 257;
  v16 = sub_B504D0(27, v4, (__int64)v14, (__int64)v31, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v16,
    v29,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v25 = *(_QWORD *)a2;
  v26 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v26 )
  {
    do
    {
      v27 = *(_QWORD *)(v25 + 8);
      v28 = *(_DWORD *)v25;
      v25 += 16;
      sub_B99FD0(v16, v28, v27);
    }
    while ( v26 != v25 );
  }
LABEL_13:
  v17 = *(_DWORD *)(v6 + 32);
  v18 = (__int64 *)sub_BCB2A0(*(_QWORD **)(a2 + 72));
  v19 = sub_BCDA70(v18, v17);
  v30 = 257;
  v20 = (__int64 **)v19;
  if ( v19 == *(_QWORD *)(v16 + 8) )
    return v16;
  v21 = *(_QWORD *)(a2 + 80);
  v22 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v21 + 120LL);
  if ( v22 == sub_920130 )
  {
    if ( *(_BYTE *)v16 > 0x15u )
    {
LABEL_24:
      v32 = 257;
      v23 = sub_B51D30(38, v16, (__int64)v20, (__int64)v31, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
        *(_QWORD *)(a2 + 88),
        v23,
        v29,
        *(_QWORD *)(a2 + 56),
        *(_QWORD *)(a2 + 64));
      sub_94AAF0((unsigned int **)a2, v23);
      return v23;
    }
    if ( (unsigned __int8)sub_AC4810(0x26u) )
      v23 = sub_ADAB70(38, v16, v20, 0);
    else
      v23 = sub_AA93C0(0x26u, v16, (__int64)v20);
  }
  else
  {
    v23 = v22(v21, 38u, (_BYTE *)v16, (__int64)v20);
  }
  if ( !v23 )
    goto LABEL_24;
  return v23;
}

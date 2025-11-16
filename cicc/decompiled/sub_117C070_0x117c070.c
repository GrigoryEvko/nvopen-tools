// Function: sub_117C070
// Address: 0x117c070
//
__int64 __fastcall sub_117C070(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned __int8 *v5; // r15
  unsigned __int8 *v6; // rbx
  unsigned int v9; // r8d
  unsigned __int8 v10; // al
  __int64 v11; // r12
  int v13; // edx
  __int64 v14; // rax
  int v15; // edi
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdi
  int v19; // r14d
  __int64 v20; // r15
  unsigned int v21; // r14d
  __int64 v22; // rdx
  int v23; // r14d
  __int64 v24; // rbx
  __int64 v25; // r13
  __int64 v26; // rdx
  unsigned int v27; // esi
  _BYTE *v28; // [rsp+10h] [rbp-F0h]
  unsigned int v29; // [rsp+18h] [rbp-E8h]
  unsigned int v30; // [rsp+1Ch] [rbp-E4h]
  __int64 v31; // [rsp+20h] [rbp-E0h]
  int v32; // [rsp+28h] [rbp-D8h]
  int v33; // [rsp+50h] [rbp-B0h] BYREF
  unsigned __int8 *v34; // [rsp+58h] [rbp-A8h]
  char v35; // [rsp+60h] [rbp-A0h]
  _DWORD v36[8]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v37; // [rsp+90h] [rbp-70h]
  _BYTE v38[32]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v39; // [rsp+C0h] [rbp-40h]

  if ( !a1 )
    return 0;
  v4 = *(_QWORD *)(a1 - 64);
  if ( !v4 )
    return 0;
  v5 = *(unsigned __int8 **)(a1 - 32);
  if ( *v5 > 0x15u )
    return 0;
  v6 = a2;
  v32 = sub_B53900(a1);
  v9 = v32;
  if ( (unsigned int)(v32 - 32) <= 1 )
    return 0;
  v10 = *a2;
  if ( *a2 <= 0x15u )
  {
    v6 = a3;
    v9 = sub_B52870(v32);
    v10 = *a3;
    a3 = a2;
    if ( v10 <= 0x1Cu )
      return 0;
  }
  else if ( v10 <= 0x1Cu )
  {
    return 0;
  }
  v13 = v10;
  if ( (unsigned int)v10 - 42 > 0x11 )
    return 0;
  if ( *a3 > 0x15u )
    return 0;
  if ( (unsigned __int8)(v10 - 51) <= 1u )
    return 0;
  if ( (unsigned int)v10 - 48 <= 1 )
    return 0;
  v14 = *((_QWORD *)v6 + 2);
  if ( !v14 )
    return 0;
  if ( *(_QWORD *)(v14 + 8) )
    return 0;
  if ( v4 != *((_QWORD *)v6 - 8) )
    return 0;
  v29 = v9;
  v28 = (_BYTE *)*((_QWORD *)v6 - 4);
  if ( *v28 > 0x15u )
    return 0;
  v30 = v13 - 29;
  v31 = sub_B43CC0((__int64)v6);
  sub_98FF80((__int64)&v33, v29, v5);
  if ( a3 == (unsigned __int8 *)sub_96E6C0(v30, (__int64)v5, v28, v31) )
  {
    v15 = sub_98FEB0(v29, 0);
  }
  else
  {
    if ( !v35 )
      return 0;
    v5 = v34;
    if ( a3 != (unsigned __int8 *)sub_96E6C0(v30, (__int64)v34, v28, v31) )
      return 0;
    v15 = sub_98FEB0(v33, 0);
  }
  v16 = sub_990550(v15);
  v36[1] = 0;
  v39 = 257;
  v17 = sub_B33C40(a4, v16, v4, (__int64)v5, v36[0], (__int64)v38);
  v18 = *(_QWORD *)(a4 + 80);
  v19 = *v6;
  v37 = 257;
  v20 = v17;
  v21 = v19 - 29;
  v11 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, _BYTE *))(*(_QWORD *)v18 + 16LL))(v18, v21, v17, v28);
  if ( !v11 )
  {
    v39 = 257;
    v11 = sub_B504D0(v21, v20, (__int64)v28, (__int64)v38, 0, 0);
    if ( (unsigned __int8)sub_920620(v11) )
    {
      v22 = *(_QWORD *)(a4 + 96);
      v23 = *(_DWORD *)(a4 + 104);
      if ( v22 )
        sub_B99FD0(v11, 3u, v22);
      sub_B45150(v11, v23);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _DWORD *, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
      *(_QWORD *)(a4 + 88),
      v11,
      v36,
      *(_QWORD *)(a4 + 56),
      *(_QWORD *)(a4 + 64));
    v24 = *(_QWORD *)a4;
    v25 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
    while ( v25 != v24 )
    {
      v26 = *(_QWORD *)(v24 + 8);
      v27 = *(_DWORD *)v24;
      v24 += 16;
      sub_B99FD0(v11, v27, v26);
    }
  }
  return v11;
}

// Function: sub_2A3A5A0
// Address: 0x2a3a5a0
//
__int64 __fastcall sub_2A3A5A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 **v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // r15
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v10; // r13
  __int64 v12; // rdx
  int v13; // r14d
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rdx
  unsigned int v17; // esi
  char v18[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v19; // [rsp+20h] [rbp-70h]
  char v20[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v21; // [rsp+50h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 32) != 3 )
  {
    v3 = *(_QWORD *)(a2 + 48);
    v4 = *(_QWORD *)(a2 + 72);
    v19 = 257;
    v5 = (__int64 **)sub_AE4420(*(_QWORD *)(*(_QWORD *)(v3 + 72) + 40LL) + 312LL, v4, 0);
    v6 = *(_QWORD *)(a2 + 48);
    v7 = *(_QWORD *)(v6 + 72);
    if ( v5 == *(__int64 ***)(v7 + 8) )
      return *(_QWORD *)(v6 + 72);
    v8 = *(_QWORD *)(a2 + 80);
    v9 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v8 + 120LL);
    if ( v9 == sub_920130 )
    {
      if ( *(_BYTE *)v7 > 0x15u )
        goto LABEL_12;
      if ( (unsigned __int8)sub_AC4810(0x2Fu) )
        v10 = sub_ADAB70(47, v7, v5, 0);
      else
        v10 = sub_AA93C0(0x2Fu, v7, (__int64)v5);
    }
    else
    {
      v10 = v9(v8, 47u, (_BYTE *)v7, (__int64)v5);
    }
    if ( v10 )
      return v10;
LABEL_12:
    v21 = 257;
    v10 = sub_B51D30(47, v7, (__int64)v5, (__int64)v20, 0, 0);
    if ( (unsigned __int8)sub_920620(v10) )
    {
      v12 = *(_QWORD *)(a2 + 96);
      v13 = *(_DWORD *)(a2 + 104);
      if ( v12 )
        sub_B99FD0(v10, 3u, v12);
      sub_B45150(v10, v13);
    }
    (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v10,
      v18,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v14 = *(_QWORD *)a2;
    v15 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    while ( v15 != v14 )
    {
      v16 = *(_QWORD *)(v14 + 8);
      v17 = *(_DWORD *)v14;
      v14 += 16;
      sub_B99FD0(v10, v17, v16);
    }
    return v10;
  }
  return sub_2A3A4E0(a2, "pc", 2u);
}

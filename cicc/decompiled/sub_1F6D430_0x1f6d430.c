// Function: sub_1F6D430
// Address: 0x1f6d430
//
__int64 __fastcall sub_1F6D430(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int16 v7; // ax
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // r14d
  int v12; // eax
  __int64 v13; // rdx
  int v14; // eax
  __int64 v15; // rax
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // eax
  __int64 (__fastcall *v22)(__int64, __int64, __int64 *, __int64, _QWORD, _QWORD, int, __int64); // r15
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // edx
  __int64 *v27; // rax
  int v28; // [rsp+0h] [rbp-60h] BYREF
  __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+10h] [rbp-50h] BYREF
  __int64 v31; // [rsp+18h] [rbp-48h]
  char v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  v7 = *(_WORD *)(a2 + 24);
  v29 = 0;
  if ( v7 == 185 )
  {
    if ( (*(_WORD *)(a2 + 26) & 0x380) != 0 || a1 != *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) )
      return 0;
  }
  else if ( v7 != 186 || (*(_WORD *)(a2 + 26) & 0x380) != 0 || a1 != *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL) )
  {
    return 0;
  }
  v9 = *(_QWORD *)(a2 + 96);
  v10 = *(_QWORD *)(a2 + 104);
  LOBYTE(v28) = *(_BYTE *)(a2 + 88);
  v29 = v9;
  v30 = 0;
  v11 = sub_1E340A0(v10);
  v12 = *(unsigned __int16 *)(a1 + 24);
  v31 = 0;
  v32 = 0;
  v33 = 0;
  if ( v12 != 52 )
  {
    if ( v12 != 53 )
      return 0;
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL);
    v14 = *(unsigned __int16 *)(v13 + 24);
    if ( v14 == 10 || v14 == 32 )
    {
      v15 = *(_QWORD *)(v13 + 88);
      v16 = *(_DWORD *)(v15 + 32);
      v17 = *(__int64 **)(v15 + 24);
      if ( v16 > 0x40 )
        v18 = *v17;
      else
        v18 = (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v16)) >> (64 - (unsigned __int8)v16);
      v19 = -v18;
      goto LABEL_15;
    }
LABEL_18:
    v33 = 1;
    goto LABEL_19;
  }
  v20 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL);
  v21 = *(unsigned __int16 *)(v20 + 24);
  if ( v21 != 32 && v21 != 10 )
    goto LABEL_18;
  v25 = *(_QWORD *)(v20 + 88);
  v26 = *(_DWORD *)(v25 + 32);
  v27 = *(__int64 **)(v25 + 24);
  if ( v26 > 0x40 )
    v19 = *v27;
  else
    v19 = (__int64)((_QWORD)v27 << (64 - (unsigned __int8)v26)) >> (64 - (unsigned __int8)v26);
LABEL_15:
  v31 = v19;
LABEL_19:
  v22 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64, _QWORD, _QWORD, int, __int64))(*(_QWORD *)a4 + 736LL);
  v23 = sub_1F58E60((__int64)&v28, *(_QWORD **)(a3 + 48));
  v24 = sub_1E0A0C0(*(_QWORD *)(a3 + 32));
  return v22(a4, v24, &v30, v23, v11, 0, v28, v29);
}

// Function: sub_26C6990
// Address: 0x26c6990
//
__int64 __fastcall sub_26C6990(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r15
  unsigned int v9; // eax
  unsigned __int8 v10; // dl
  __int64 v11; // r15
  int v12; // edx
  _QWORD *v13; // rdi
  __int64 v14; // rax
  unsigned int *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 (__fastcall **v19)(); // rax
  __int64 (__fastcall **v21)(); // rax
  int v22; // edx
  __int64 v23; // r15
  unsigned int v24; // ecx
  __int64 *v25; // r13
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // [rsp+8h] [rbp-228h] BYREF
  unsigned int v31; // [rsp+Ch] [rbp-224h] BYREF
  _QWORD v32[2]; // [rsp+10h] [rbp-220h] BYREF
  char v33; // [rsp+20h] [rbp-210h]
  _QWORD v34[4]; // [rsp+30h] [rbp-200h] BYREF
  _QWORD v35[10]; // [rsp+50h] [rbp-1E0h] BYREF
  _BYTE v36[400]; // [rsp+A0h] [rbp-190h] BYREF

  v6 = (*(__int64 (__fastcall **)(_QWORD *, __int64))(*a2 + 16LL))(a2, a3);
  if ( !v6 || !*(_QWORD *)(a3 + 48) )
  {
    v19 = sub_2241E40();
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v19;
    return a1;
  }
  v7 = v6;
  v8 = sub_B10CD0(a3 + 48);
  v9 = sub_C1B040(v8);
  v30 = v9;
  v10 = *(_BYTE *)(v8 - 16);
  if ( LOBYTE(qword_4F813A8[8]) )
  {
    if ( (v10 & 2) != 0 )
      v11 = *(_QWORD *)(v8 - 32);
    else
      v11 = v8 - 16 - 8LL * ((v10 >> 2) & 0xF);
    v12 = 0;
    if ( **(_BYTE **)v11 == 20 )
      v12 = *(_DWORD *)(*(_QWORD *)v11 + 4LL);
  }
  else
  {
    if ( (v10 & 2) != 0 )
      v23 = *(_QWORD *)(v8 - 32);
    else
      v23 = v8 - 16 - 8LL * ((v10 >> 2) & 0xF);
    v12 = 0;
    if ( **(_BYTE **)v23 == 20 )
    {
      v24 = *(_DWORD *)(*(_QWORD *)v23 + 4LL);
      if ( (v24 & 7) == 7 && (v24 & 0xFFFFFFF8) != 0 )
      {
        if ( (v24 & 0x10000000) != 0 )
          v12 = BYTE2(v24) & 7;
        else
          v12 = (unsigned __int16)(v24 >> 3);
      }
      else if ( (v24 & 1) != 0 )
      {
        v12 = 0;
      }
      else
      {
        v12 = (v24 >> 1) & 0x1F;
        if ( ((v24 >> 1) & 0x20) != 0 )
          v12 |= (v24 >> 2) & 0xFE0;
      }
    }
  }
  v35[0] = __PAIR64__(v12, v9);
  v13 = *(_QWORD **)(v7 + 168);
  v31 = v12;
  if ( v13 && (v14 = sub_C1BA30(v13, (__int64)v35)) != 0 )
    v15 = (unsigned int *)(v14 + 16);
  else
    v15 = (unsigned int *)v35;
  v16 = sub_26C2A80(v7 + 72, v15);
  if ( v16 == v7 + 80 )
  {
    v21 = sub_2241E40();
    v22 = 0;
LABEL_20:
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v22;
    *(_QWORD *)(a1 + 8) = v21;
    return a1;
  }
  v17 = *(_QWORD *)(v16 + 40);
  v33 &= ~1u;
  v32[0] = v17;
  if ( (unsigned __int8)sub_2A61A10(a2 + 136, v7, v30, v31) )
  {
    v25 = (__int64 *)a2[161];
    v34[0] = a3;
    v34[1] = v32;
    v34[2] = &v30;
    v34[3] = &v31;
    v26 = *v25;
    v27 = sub_B2BE50(*v25);
    if ( sub_B6EA50(v27)
      || (v28 = sub_B2BE50(v26),
          v29 = sub_B6F970(v28),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v29 + 48LL))(v29)) )
    {
      sub_26C6600((__int64)v35, (__int64)v34);
      sub_1049740(v25, (__int64)v35);
      v35[0] = &unk_49D9D40;
      sub_23FD590((__int64)v36);
    }
  }
  if ( (v33 & 1) != 0 )
  {
    v22 = v32[0];
    v21 = (__int64 (__fastcall **)())v32[1];
    goto LABEL_20;
  }
  v18 = v32[0];
  *(_BYTE *)(a1 + 16) &= ~1u;
  *(_QWORD *)a1 = v18;
  return a1;
}

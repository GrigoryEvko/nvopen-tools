// Function: sub_1706EB0
// Address: 0x1706eb0
//
__int64 __fastcall sub_1706EB0(__int64 a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  int v8; // edi
  __int64 **v9; // rdx
  int v10; // edi
  unsigned __int8 *v11; // r12
  __int64 v12; // rax
  __int64 v14; // rdx
  unsigned __int8 v15; // cl
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 *v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned __int64 *v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rdx
  bool v26; // zf
  __int64 v27; // rsi
  __int64 v28; // rsi
  unsigned __int8 *v29; // rsi
  __int64 v30; // r15
  const char *v31; // rax
  int v32; // esi
  __int64 v33; // rdx
  unsigned __int8 *v34; // rax
  unsigned __int8 v35; // dl
  __int64 v36; // rcx
  char v37; // al
  unsigned __int8 *v38; // [rsp+8h] [rbp-68h] BYREF
  __int64 v39[2]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v40; // [rsp+20h] [rbp-50h]
  __int64 v41[2]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v42; // [rsp+40h] [rbp-30h]

  v8 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned int)(v8 - 60) <= 0xC )
  {
    v9 = *(__int64 ***)a1;
    v10 = v8 - 24;
    v40 = 257;
    if ( v9 == *(__int64 ***)a2 )
      return a2;
    if ( *(_BYTE *)(a2 + 16) > 0x10u )
    {
      v42 = 257;
      v20 = sub_15FDBD0(v10, a2, (__int64)v9, (__int64)v41, 0);
      v21 = *(_QWORD *)(a3 + 8);
      v11 = (unsigned __int8 *)v20;
      if ( v21 )
      {
        v22 = *(unsigned __int64 **)(a3 + 16);
        sub_157E9D0(v21 + 40, v20);
        v23 = *((_QWORD *)v11 + 3);
        v24 = *v22;
        *((_QWORD *)v11 + 4) = v22;
        v24 &= 0xFFFFFFFFFFFFFFF8LL;
        *((_QWORD *)v11 + 3) = v24 | v23 & 7;
        *(_QWORD *)(v24 + 8) = v11 + 24;
        *v22 = *v22 & 7 | (unsigned __int64)(v11 + 24);
      }
      sub_164B780((__int64)v11, v39);
      v26 = *(_QWORD *)(a3 + 80) == 0;
      v38 = v11;
      if ( v26 )
        sub_4263D6(v11, v39, v25);
      (*(void (__fastcall **)(__int64, unsigned __int8 **))(a3 + 88))(a3 + 64, &v38);
      v27 = *(_QWORD *)a3;
      if ( *(_QWORD *)a3 )
      {
        v38 = *(unsigned __int8 **)a3;
        sub_1623A60((__int64)&v38, v27, 2);
        v28 = *((_QWORD *)v11 + 6);
        if ( v28 )
          sub_161E7C0((__int64)(v11 + 48), v28);
        v29 = v38;
        *((_QWORD *)v11 + 6) = v38;
        if ( v29 )
          sub_1623210((__int64)&v38, v29, (__int64)(v11 + 48));
      }
    }
    else
    {
      v11 = (unsigned __int8 *)sub_15A46C0(v10, (__int64 ***)a2, v9, 0);
      v12 = sub_14DBA30((__int64)v11, *(_QWORD *)(a3 + 96), 0);
      if ( v12 )
        return v12;
    }
    return (__int64)v11;
  }
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v14 = *(_QWORD *)(a1 - 8);
  else
    v14 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v15 = *(_BYTE *)(*(_QWORD *)(v14 + 24) + 16LL);
  v16 = v15 < 0x11u ? 0x18 : 0;
  v17 = *(_QWORD *)(v14 + v16);
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
    v30 = a2;
    if ( v15 > 0x10u )
    {
      v30 = *(_QWORD *)(v14 + v16);
      v17 = a2;
    }
    v31 = sub_1649960(a2);
    v32 = *(unsigned __int8 *)(a1 + 16);
    v39[0] = (__int64)v31;
    v42 = 773;
    v39[1] = v33;
    v41[0] = (__int64)v39;
    v41[1] = (__int64)".op";
    v34 = (unsigned __int8 *)sub_17066B0(a3, v32 - 24, v30, v17, v41, 0, a4, a5, a6);
    v35 = v34[16];
    v11 = v34;
    if ( v35 > 0x17u )
    {
      v36 = *(_QWORD *)v34;
      v37 = *(_BYTE *)(*(_QWORD *)v34 + 8LL);
      if ( v37 == 16 )
        v37 = *(_BYTE *)(**(_QWORD **)(v36 + 16) + 8LL);
      if ( (unsigned __int8)(v37 - 1) <= 5u || v35 == 76 )
        sub_15F2500((__int64)v11, a1);
    }
    return (__int64)v11;
  }
  v18 = (__int64 *)(unsigned int)(v8 - 24);
  v19 = *(_QWORD *)(v14 + v16);
  if ( v15 > 0x10u )
  {
    v19 = a2;
    a2 = v17;
  }
  return sub_15A2A30(v18, (__int64 *)a2, v19, 0, 0, a4, a5, a6);
}

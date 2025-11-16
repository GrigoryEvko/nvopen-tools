// Function: sub_17CF280
// Address: 0x17cf280
//
unsigned __int64 __fastcall sub_17CF280(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 **v11; // rsi
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int64 result; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 *v17; // r15
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int8 *v22; // rsi
  _QWORD *v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  unsigned __int64 v27; // [rsp-10h] [rbp-160h]
  __int64 *v28; // [rsp+8h] [rbp-148h]
  __int64 v29; // [rsp+10h] [rbp-140h] BYREF
  __int16 v30; // [rsp+20h] [rbp-130h]
  char v31[16]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v32; // [rsp+40h] [rbp-110h]
  __int64 v33; // [rsp+50h] [rbp-100h] BYREF
  __int16 v34; // [rsp+60h] [rbp-F0h]
  _QWORD v35[4]; // [rsp+70h] [rbp-E0h] BYREF
  char v36[16]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v37; // [rsp+A0h] [rbp-B0h]
  _BYTE v38[16]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v39; // [rsp+C0h] [rbp-90h]
  __int64 v40; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v41; // [rsp+D8h] [rbp-78h]
  __int64 *v42; // [rsp+E0h] [rbp-70h]
  _QWORD *v43; // [rsp+E8h] [rbp-68h]

  sub_17CE510((__int64)&v40, a2, 0, 0, 0);
  v37 = 257;
  v30 = 257;
  v4 = sub_16471D0(v43, 0);
  v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( v4 != *(_QWORD *)v5 )
  {
    if ( *(_BYTE *)(v5 + 16) > 0x10u )
    {
      v15 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v39 = 257;
      v16 = sub_15FDFF0(v15, v4, (__int64)v38, 0);
      v5 = v16;
      if ( v41 )
      {
        v17 = v42;
        sub_157E9D0(v41 + 40, v16);
        v18 = *(_QWORD *)(v5 + 24);
        v19 = *v17;
        *(_QWORD *)(v5 + 32) = v17;
        v19 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v5 + 24) = v19 | v18 & 7;
        *(_QWORD *)(v19 + 8) = v5 + 24;
        *v17 = *v17 & 7 | (v5 + 24);
      }
      sub_164B780(v5, &v29);
      if ( v40 )
      {
        v35[0] = v40;
        sub_1623A60((__int64)v35, v40, 2);
        v20 = *(_QWORD *)(v5 + 48);
        v21 = v5 + 48;
        if ( v20 )
        {
          sub_161E7C0(v5 + 48, v20);
          v21 = v5 + 48;
        }
        v22 = (unsigned __int8 *)v35[0];
        *(_QWORD *)(v5 + 48) = v35[0];
        if ( v22 )
          sub_1623210((__int64)v35, v22, v21);
      }
    }
    else
    {
      v5 = sub_15A4A70(*(__int64 ****)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v4);
    }
  }
  v35[0] = v5;
  v32 = 257;
  v6 = sub_16471D0(v43, 0);
  v7 = sub_12A95D0(&v40, *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))), v6, (__int64)v31);
  v8 = *(_DWORD *)(a2 + 20);
  v9 = *(_QWORD *)(a1 + 8);
  v35[1] = v7;
  v10 = v8 & 0xFFFFFFF;
  v34 = 257;
  v11 = *(__int64 ***)(v9 + 176);
  v12 = *(_QWORD *)(a2 + 24 * (2 - v10));
  if ( v11 != *(__int64 ***)v12 )
  {
    if ( *(_BYTE *)(v12 + 16) > 0x10u )
    {
      v23 = *(_QWORD **)(a2 + 24 * (2 - v10));
      v39 = 257;
      v24 = sub_15FE0A0(v23, (__int64)v11, 0, (__int64)v38, 0);
      v12 = v24;
      if ( v41 )
      {
        v28 = v42;
        sub_157E9D0(v41 + 40, v24);
        v25 = *v28;
        v26 = *(_QWORD *)(v12 + 24) & 7LL;
        *(_QWORD *)(v12 + 32) = v28;
        v25 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v12 + 24) = v25 | v26;
        *(_QWORD *)(v25 + 8) = v12 + 24;
        *v28 = *v28 & 7 | (v12 + 24);
      }
      sub_164B780(v12, &v33);
      sub_12A86E0(&v40, v12);
      v9 = *(_QWORD *)(a1 + 8);
    }
    else
    {
      v13 = sub_15A4750(*(__int64 ****)(a2 + 24 * (2 - v10)), v11, 0);
      v9 = *(_QWORD *)(a1 + 8);
      v12 = v13;
    }
  }
  v35[2] = v12;
  sub_1285290(&v40, *(_QWORD *)(**(_QWORD **)(v9 + 352) + 24LL), *(_QWORD *)(v9 + 352), (int)v35, 3, (__int64)v36, 0);
  sub_15F20C0((_QWORD *)a2);
  result = v27;
  if ( v40 )
    return sub_161E7C0((__int64)&v40, v40);
  return result;
}

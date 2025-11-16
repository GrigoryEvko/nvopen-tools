// Function: sub_17CEF70
// Address: 0x17cef70
//
unsigned __int64 __fastcall sub_17CEF70(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 **v10; // rsi
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 result; // rax
  __int64 v15; // rax
  __int64 *v16; // r15
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rdx
  unsigned __int8 *v21; // rsi
  _QWORD *v22; // rdi
  __int64 v23; // rax
  __int64 *v24; // r15
  __int64 v25; // rax
  __int64 v26; // rcx
  unsigned __int64 v27; // [rsp-10h] [rbp-160h]
  char v28[16]; // [rsp+10h] [rbp-140h] BYREF
  __int16 v29; // [rsp+20h] [rbp-130h]
  __int64 v30; // [rsp+30h] [rbp-120h] BYREF
  __int16 v31; // [rsp+40h] [rbp-110h]
  __int64 v32[2]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v33; // [rsp+60h] [rbp-F0h]
  _QWORD v34[4]; // [rsp+70h] [rbp-E0h] BYREF
  char v35[16]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v36; // [rsp+A0h] [rbp-B0h]
  _BYTE v37[16]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v38; // [rsp+C0h] [rbp-90h]
  __int64 v39; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v40; // [rsp+D8h] [rbp-78h]
  __int64 *v41; // [rsp+E0h] [rbp-70h]
  _QWORD *v42; // [rsp+E8h] [rbp-68h]

  sub_17CE510((__int64)&v39, a2, 0, 0, 0);
  v36 = 257;
  v29 = 257;
  v4 = sub_16471D0(v42, 0);
  v34[0] = sub_12A95D0(&v39, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)), v4, (__int64)v28);
  v31 = 257;
  v5 = sub_16471D0(v42, 0);
  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v7 = *(_QWORD *)(a2 + 24 * (1 - v6));
  if ( v5 != *(_QWORD *)v7 )
  {
    if ( *(_BYTE *)(v7 + 16) > 0x10u )
    {
      v38 = 257;
      v15 = sub_15FDFF0(v7, v5, (__int64)v37, 0);
      v7 = v15;
      if ( v40 )
      {
        v16 = v41;
        sub_157E9D0(v40 + 40, v15);
        v17 = *(_QWORD *)(v7 + 24);
        v18 = *v16;
        *(_QWORD *)(v7 + 32) = v16;
        v18 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v7 + 24) = v18 | v17 & 7;
        *(_QWORD *)(v18 + 8) = v7 + 24;
        *v16 = *v16 & 7 | (v7 + 24);
      }
      sub_164B780(v7, &v30);
      if ( v39 )
      {
        v32[0] = v39;
        sub_1623A60((__int64)v32, v39, 2);
        v19 = *(_QWORD *)(v7 + 48);
        v20 = v7 + 48;
        if ( v19 )
        {
          sub_161E7C0(v7 + 48, v19);
          v20 = v7 + 48;
        }
        v21 = (unsigned __int8 *)v32[0];
        *(_QWORD *)(v7 + 48) = v32[0];
        if ( v21 )
          sub_1623210((__int64)v32, v21, v20);
      }
      v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
    else
    {
      v7 = sub_15A4A70(*(__int64 ****)(a2 + 24 * (1 - v6)), v5);
      v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
  }
  v34[1] = v7;
  v8 = *(_QWORD *)(a1 + 8);
  v33 = 257;
  v9 = 3 * (2 - v6);
  v10 = *(__int64 ***)(v8 + 176);
  v11 = *(_QWORD *)(a2 + 8 * v9);
  if ( v10 != *(__int64 ***)v11 )
  {
    if ( *(_BYTE *)(v11 + 16) > 0x10u )
    {
      v22 = *(_QWORD **)(a2 + 8 * v9);
      v38 = 257;
      v23 = sub_15FE0A0(v22, (__int64)v10, 0, (__int64)v37, 0);
      v11 = v23;
      if ( v40 )
      {
        v24 = v41;
        sub_157E9D0(v40 + 40, v23);
        v25 = *(_QWORD *)(v11 + 24);
        v26 = *v24;
        *(_QWORD *)(v11 + 32) = v24;
        v26 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v11 + 24) = v26 | v25 & 7;
        *(_QWORD *)(v26 + 8) = v11 + 24;
        *v24 = *v24 & 7 | (v11 + 24);
      }
      sub_164B780(v11, v32);
      sub_12A86E0(&v39, v11);
      v8 = *(_QWORD *)(a1 + 8);
    }
    else
    {
      v12 = sub_15A4750(*(__int64 ****)(a2 + 8 * v9), v10, 0);
      v8 = *(_QWORD *)(a1 + 8);
      v11 = v12;
    }
  }
  v13 = *(_QWORD *)(v8 + 360);
  v34[2] = v11;
  sub_1285290(&v39, *(_QWORD *)(*(_QWORD *)v13 + 24LL), v13, (int)v34, 3, (__int64)v35, 0);
  sub_15F20C0((_QWORD *)a2);
  result = v27;
  if ( v39 )
    return sub_161E7C0((__int64)&v39, v39);
  return result;
}

// Function: sub_17E2B70
// Address: 0x17e2b70
//
_DWORD *__fastcall sub_17E2B70(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 **v9; // r12
  __int64 ***v10; // rdi
  unsigned __int8 *v11; // rax
  __int64 v12; // r12
  __int64 v13; // rax
  unsigned __int8 *v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rax
  _DWORD *result; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  unsigned __int8 *v26; // rsi
  __int64 **v27; // [rsp+10h] [rbp-F0h]
  __int64 *v28; // [rsp+10h] [rbp-F0h]
  __int64 *v29; // [rsp+18h] [rbp-E8h]
  unsigned __int8 *v30; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v31[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int16 v32; // [rsp+40h] [rbp-C0h]
  unsigned __int8 *v33[2]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+60h] [rbp-A0h]
  __int64 v35; // [rsp+68h] [rbp-98h]
  __int64 v36; // [rsp+70h] [rbp-90h]
  unsigned __int8 *v37; // [rsp+80h] [rbp-80h] BYREF
  __int64 v38; // [rsp+88h] [rbp-78h]
  __int64 *v39; // [rsp+90h] [rbp-70h]
  _QWORD *v40; // [rsp+98h] [rbp-68h]
  __int64 v41; // [rsp+A0h] [rbp-60h]
  int v42; // [rsp+A8h] [rbp-58h]
  __int64 v43; // [rsp+B0h] [rbp-50h]
  __int64 v44; // [rsp+B8h] [rbp-48h]

  v29 = *(__int64 **)(*(_QWORD *)a1 + 40LL);
  v4 = (_QWORD *)sub_16498A0(a2);
  v5 = *(unsigned __int8 **)(a2 + 48);
  v37 = 0;
  v40 = v4;
  v6 = *(_QWORD *)(a2 + 40);
  v41 = 0;
  v38 = v6;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v39 = (__int64 *)(a2 + 24);
  v33[0] = v5;
  if ( v5 )
  {
    sub_1623A60((__int64)v33, (__int64)v5, 2);
    if ( v37 )
      sub_161E7C0((__int64)&v37, (__int64)v37);
    v37 = v33[0];
    if ( v33[0] )
      sub_1623210((__int64)v33, v33[0], (__int64)&v37);
  }
  v27 = (__int64 **)sub_1643360(v40);
  v7 = sub_16471D0(v40, 0);
  v8 = *(_QWORD *)(a2 - 72);
  v9 = (__int64 **)v7;
  v32 = 257;
  if ( v27 != *(__int64 ***)v8 )
  {
    if ( *(_BYTE *)(v8 + 16) > 0x10u )
    {
      LOWORD(v34) = 257;
      v21 = sub_15FDBD0(37, v8, (__int64)v27, (__int64)v33, 0);
      v8 = v21;
      if ( v38 )
      {
        v28 = v39;
        sub_157E9D0(v38 + 40, v21);
        v22 = *v28;
        v23 = *(_QWORD *)(v8 + 24) & 7LL;
        *(_QWORD *)(v8 + 32) = v28;
        v22 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v22 | v23;
        *(_QWORD *)(v22 + 8) = v8 + 24;
        *v28 = *v28 & 7 | (v8 + 24);
      }
      sub_164B780(v8, v31);
      if ( v37 )
      {
        v30 = v37;
        sub_1623A60((__int64)&v30, (__int64)v37, 2);
        v24 = *(_QWORD *)(v8 + 48);
        v25 = v8 + 48;
        if ( v24 )
        {
          sub_161E7C0(v8 + 48, v24);
          v25 = v8 + 48;
        }
        v26 = v30;
        *(_QWORD *)(v8 + 48) = v30;
        if ( v26 )
          sub_1623210((__int64)&v30, v26, v25);
      }
    }
    else
    {
      v8 = sub_15A46C0(37, (__int64 ***)v8, v27, 0);
    }
  }
  v10 = *(__int64 ****)(a1 + 32);
  v32 = 257;
  v11 = (unsigned __int8 *)sub_15A4510(v10, v9, 0);
  v12 = *(_QWORD *)(a1 + 40);
  v33[0] = v11;
  v13 = sub_1643360(v40);
  v14 = (unsigned __int8 *)sub_159C470(v13, v12, 0);
  v15 = *(unsigned int *)(a1 + 24);
  v33[1] = v14;
  v16 = sub_1643350(v40);
  v34 = sub_159C470(v16, v15, 0);
  v17 = **(unsigned int **)(a1 + 16);
  v18 = sub_1643350(v40);
  v35 = sub_159C470(v18, v17, 0);
  v36 = v8;
  v19 = sub_15E26F0(v29, 111, 0, 0);
  sub_17E28C0((__int64)&v37, *(_QWORD *)(v19 + 24), v19, (__int64 *)v33, 5, v31, 0);
  result = *(_DWORD **)(a1 + 16);
  ++*result;
  if ( v37 )
    return (_DWORD *)sub_161E7C0((__int64)&v37, (__int64)v37);
  return result;
}

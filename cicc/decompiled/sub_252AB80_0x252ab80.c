// Function: sub_252AB80
// Address: 0x252ab80
//
__int64 __fastcall sub_252AB80(__int64 *a1, unsigned __int8 *a2)
{
  unsigned int v2; // r13d
  int v3; // edx
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // r13d
  __int64 v20; // rax
  __int64 v21; // r13
  _BYTE *v22; // r15
  __int64 v23; // r14
  __m128i v24; // rax
  unsigned __int64 v25; // rax
  __int64 *v26; // rdx
  char v27; // [rsp+5h] [rbp-BBh] BYREF
  char v28; // [rsp+6h] [rbp-BAh] BYREF
  char v29; // [rsp+7h] [rbp-B9h] BYREF
  __int64 v30; // [rsp+8h] [rbp-B8h] BYREF
  _QWORD v31[2]; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD v32[2]; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int64 v33; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v34; // [rsp+38h] [rbp-88h]
  __int64 v35; // [rsp+40h] [rbp-80h]
  __int64 v36; // [rsp+48h] [rbp-78h]
  __m128i v37; // [rsp+50h] [rbp-70h] BYREF
  char *v38; // [rsp+60h] [rbp-60h]
  char *v39; // [rsp+68h] [rbp-58h]
  __int64 *v40; // [rsp+70h] [rbp-50h]
  __int64 v41; // [rsp+78h] [rbp-48h]
  unsigned __int64 v42; // [rsp+80h] [rbp-40h]
  __int64 v43; // [rsp+88h] [rbp-38h]

  v2 = 1;
  v3 = *a2;
  if ( (unsigned int)(v3 - 12) <= 1 )
    return v2;
  if ( (_BYTE)v3 != 20 )
  {
    if ( (_BYTE)v3 != 60 )
    {
      if ( (_BYTE)v3 == 3 )
        goto LABEL_26;
      if ( !(unsigned __int8)sub_CF6FD0(a2) )
        return 0;
      if ( *a2 == 3 )
      {
LABEL_26:
        if ( (a2[32] & 0xFu) - 7 > 1 && ((a2[80] & 1) == 0 || sub_B2FC80((__int64)a2)) )
          return 0;
      }
    }
    v5 = *a1;
    v6 = a1[7];
    v7 = a1[6];
    v37.m128i_i64[0] = (__int64)v31;
    v8 = a1[9];
    v30 = v5;
    v9 = a1[8];
    v31[1] = &v28;
    v34 = &v30;
    v10 = a1[2];
    v39 = &v28;
    v40 = &v30;
    v31[0] = &v27;
    v35 = v6;
    v37.m128i_i64[1] = v8;
    v38 = &v27;
    v41 = v6;
    v27 = 1;
    v28 = 0;
    v33 = v7;
    v36 = v9;
    v42 = v7;
    v43 = v9;
    v29 = 0;
    v32[0] = 0xFFFFFFFF80000000LL;
    v32[1] = 0xFFFFFFFF80000000LL;
    v11 = sub_250D2C0((unsigned __int64)a2, 0);
    v13 = sub_252A820(v10, v11, v12, a1[3], 2, 0, 1);
    if ( v13 )
    {
      v2 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, __int64, __int64 (__fastcall *)(), __m128i *, char *, _QWORD *, __int64 (__fastcall *)(__int64, __int64), unsigned __int64 *))(*(_QWORD *)v13 + 160LL))(
             v13,
             a1[2],
             a1[3],
             *a1,
             0,
             1,
             sub_252F350,
             &v37,
             &v29,
             v32,
             sub_2505F30,
             &v33);
      if ( (_BYTE)v2 )
      {
        v16 = a1[10];
        v17 = *(unsigned int *)(v16 + 8);
        if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 12) )
        {
          sub_C8D5F0(v16, (const void *)(v16 + 16), v17 + 1, 8u, v14, v15);
          v17 = *(unsigned int *)(v16 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v16 + 8 * v17) = v13;
        ++*(_DWORD *)(v16 + 8);
        return v2;
      }
    }
    return 0;
  }
  v18 = *(_QWORD *)(a1[1] + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
    v18 = **(_QWORD **)(v18 + 16);
  v19 = *(_DWORD *)(v18 + 8);
  v20 = sub_B43CB0(*a1);
  if ( sub_B2F070(v20, v19 >> 8) )
    return 0;
  v21 = a1[2];
  v22 = (_BYTE *)a1[4];
  v23 = a1[3];
  v24.m128i_i64[0] = sub_250D2C0(a1[1], 0);
  v37 = v24;
  v25 = sub_2527850(v21, &v37, v23, v22, 2u);
  v33 = v25;
  v34 = v26;
  v2 = (unsigned __int8)v26;
  if ( !(_BYTE)v26 || a2 != (unsigned __int8 *)v25 )
    return 0;
  return v2;
}

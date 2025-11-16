// Function: sub_252ED40
// Address: 0x252ed40
//
__int64 __fastcall sub_252ED40(__int64 *a1, unsigned __int8 *a2)
{
  unsigned int v2; // r13d
  int v3; // edx
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // rdx
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // r13
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned int v21; // r13d
  __int64 v22; // rax
  __int64 v23; // r13
  _BYTE *v24; // r15
  __int64 v25; // r14
  __m128i v26; // rax
  unsigned __int64 v27; // rax
  __int64 *v28; // rdx
  unsigned __int8 *v29; // rax
  int v30; // ecx
  __int64 v31; // rdi
  char v32; // [rsp+5h] [rbp-CBh] BYREF
  char v33; // [rsp+6h] [rbp-CAh] BYREF
  char v34; // [rsp+7h] [rbp-C9h] BYREF
  __int64 v35; // [rsp+8h] [rbp-C8h] BYREF
  unsigned __int8 *v36; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+18h] [rbp-B8h] BYREF
  char *v38; // [rsp+20h] [rbp-B0h] BYREF
  char *v39; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v40[2]; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int64 v41; // [rsp+40h] [rbp-90h] BYREF
  __int64 *v42; // [rsp+48h] [rbp-88h]
  __int64 v43; // [rsp+50h] [rbp-80h]
  __int64 v44; // [rsp+58h] [rbp-78h]
  __m128i v45; // [rsp+60h] [rbp-70h] BYREF
  char *v46; // [rsp+70h] [rbp-60h]
  char *v47; // [rsp+78h] [rbp-58h]
  __int64 *v48; // [rsp+80h] [rbp-50h]
  __int64 v49; // [rsp+88h] [rbp-48h]
  unsigned __int64 v50; // [rsp+90h] [rbp-40h]
  __int64 v51; // [rsp+98h] [rbp-38h]

  v2 = 1;
  v3 = *a2;
  if ( (unsigned int)(v3 - 12) <= 1 )
    return v2;
  if ( (_BYTE)v3 == 20 )
  {
    v20 = *(_QWORD *)(a1[1] + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
      v20 = **(_QWORD **)(v20 + 16);
    v21 = *(_DWORD *)(v20 + 8);
    v22 = sub_B43CB0(*a1);
    if ( sub_B2F070(v22, v21 >> 8) )
      return 0;
    v23 = a1[2];
    v24 = (_BYTE *)a1[4];
    v25 = a1[3];
    v26.m128i_i64[0] = sub_250D2C0(a1[1], 0);
    v45 = v26;
    v27 = sub_2527850(v23, &v45, v25, v24, 2u);
    v41 = v27;
    v42 = v28;
    v2 = (unsigned __int8)v28;
    if ( !(_BYTE)v28 || a2 != (unsigned __int8 *)v27 )
      return 0;
    return v2;
  }
  if ( (_BYTE)v3 != 60 )
  {
    if ( (_BYTE)v3 == 3 )
      goto LABEL_38;
    if ( !sub_D5CAE0(a2, *(__int64 **)a1[5]) )
      return 0;
    if ( *a2 == 3 )
    {
LABEL_38:
      if ( (a2[32] & 0xFu) - 7 > 1 && ((a2[80] & 1) == 0 || sub_B2FC80((__int64)a2)) )
        return 0;
    }
  }
  v6 = *a1;
  v7 = a1[7];
  v8 = a1[6];
  v45.m128i_i64[0] = (__int64)&v38;
  v9 = a1[9];
  v35 = v6;
  v10 = a1[8];
  v39 = &v33;
  v42 = &v35;
  v11 = a1[2];
  v47 = &v33;
  v48 = &v35;
  v38 = &v32;
  v43 = v7;
  v45.m128i_i64[1] = v9;
  v46 = &v32;
  v49 = v7;
  v32 = 1;
  v33 = 0;
  v41 = v8;
  v44 = v10;
  v50 = v8;
  v51 = v10;
  v34 = 0;
  v40[0] = 0xFFFFFFFF80000000LL;
  v40[1] = 0xFFFFFFFF80000000LL;
  v12 = sub_250D2C0((unsigned __int64)a2, 0);
  v14 = sub_252A820(v11, v12, v13, a1[3], 2, 0, 1);
  v15 = v14;
  if ( !v14 )
    return 0;
  v2 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, _QWORD, __int64 (__fastcall *)(), __m128i *, char *, unsigned __int64 *, __int64 (__fastcall *)(__int64, __int64), unsigned __int64 *))(*(_QWORD *)v14 + 160LL))(
         v14,
         a1[2],
         a1[3],
         *a1,
         1,
         0,
         sub_252F120,
         &v45,
         &v34,
         v40,
         sub_252E6C0,
         &v41);
  if ( !(_BYTE)v2 )
    return 0;
  if ( !v34 && v40[0] != 0xFFFFFFFF80000000LL )
  {
    v29 = (unsigned __int8 *)sub_2511870(
                               a1[2],
                               a1[3],
                               a2,
                               *(__int64 ***)(*a1 + 8),
                               *(__int64 **)a1[5],
                               *(_BYTE **)(*(_QWORD *)(a1[2] + 208) + 104LL),
                               v40);
    v36 = v29;
    if ( !v29 )
      return 0;
    v30 = *v29;
    if ( (unsigned int)(v30 - 12) > 1 )
    {
      if ( (unsigned __int8)v30 <= 0x15u && sub_AC30F0((__int64)v29) )
        *v39 = 0;
      else
        *v38 = 0;
    }
    if ( v33 && !v32 )
      return 0;
    sub_252E900(a1[7], (__int64 *)&v36);
    if ( *(_QWORD *)a1[6] )
    {
      v31 = a1[8];
      v37 = 0;
      sub_252E280(v31, &v37);
    }
  }
  v18 = a1[10];
  v19 = *(unsigned int *)(v18 + 8);
  if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
  {
    sub_C8D5F0(v18, (const void *)(v18 + 16), v19 + 1, 8u, v16, v17);
    v19 = *(unsigned int *)(v18 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v18 + 8 * v19) = v15;
  ++*(_DWORD *)(v18 + 8);
  return v2;
}

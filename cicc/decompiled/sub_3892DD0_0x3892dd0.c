// Function: sub_3892DD0
// Address: 0x3892dd0
//
__int64 __fastcall sub_3892DD0(__int64 a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // r12
  int v6; // eax
  unsigned int v7; // r12d
  __int64 v9; // rdi
  __int64 v10; // rcx
  unsigned int v11; // eax
  unsigned __int64 *v12; // rdx
  __int64 v13; // rax
  _BYTE *v14; // rsi
  const char *v15; // rax
  int v16; // eax
  unsigned __int64 v17; // rax
  __int64 v18; // r8
  unsigned int v19; // eax
  unsigned __int64 *v20; // rdx
  __int64 v21; // rax
  _BYTE *v22; // rsi
  int v23; // eax
  __int64 v24; // [rsp+8h] [rbp-118h]
  __int64 v25; // [rsp+10h] [rbp-110h]
  unsigned __int64 v26; // [rsp+28h] [rbp-F8h]
  __int64 v28; // [rsp+48h] [rbp-D8h] BYREF
  _QWORD v29[2]; // [rsp+50h] [rbp-D0h] BYREF
  char v30; // [rsp+60h] [rbp-C0h]
  char v31; // [rsp+61h] [rbp-BFh]
  _QWORD *v32; // [rsp+70h] [rbp-B0h] BYREF
  unsigned __int64 v33; // [rsp+78h] [rbp-A8h]
  _BYTE v34[16]; // [rsp+80h] [rbp-A0h] BYREF
  __m128i v35; // [rsp+90h] [rbp-90h] BYREF
  int v36; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD *v37; // [rsp+A8h] [rbp-78h]
  int *v38; // [rsp+B0h] [rbp-70h]
  int *v39; // [rsp+B8h] [rbp-68h]
  __int64 v40; // [rsp+C0h] [rbp-60h]
  __int64 v41; // [rsp+C8h] [rbp-58h]
  __int64 v42; // [rsp+D0h] [rbp-50h]
  __int64 v43; // [rsp+D8h] [rbp-48h]
  __int64 v44; // [rsp+E0h] [rbp-40h]
  __int64 v45; // [rsp+E8h] [rbp-38h]

  v4 = a1 + 8;
  *a3 = 0;
  v6 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v6;
  if ( v6 == 13 )
    return sub_388AF10(a1, 13, "expected ')' at end of argument list");
  if ( v6 == 2 )
  {
    *a3 = 1;
    *(_DWORD *)(a1 + 64) = sub_3887100(v4);
    return sub_388AF10(a1, 13, "expected ')' at end of argument list");
  }
  v26 = *(_QWORD *)(a1 + 56);
  v38 = &v36;
  v39 = &v36;
  v29[0] = "expected type";
  v28 = 0;
  v35.m128i_i64[0] = 0;
  v36 = 0;
  v37 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v32 = v34;
  v33 = 0;
  v34[0] = 0;
  v31 = 1;
  v30 = 3;
  if ( (unsigned __int8)sub_3891B00(a1, &v28, (__int64)v29, 0) || (unsigned __int8)sub_388C730(a1, &v35) )
  {
LABEL_4:
    v7 = 1;
  }
  else
  {
    v9 = v28;
    if ( *(_BYTE *)(v28 + 8) )
    {
      if ( *(_DWORD *)(a1 + 64) == 375 )
      {
        sub_2240AE0((unsigned __int64 *)&v32, (unsigned __int64 *)(a1 + 72));
        v23 = sub_3887100(v4);
        v9 = v28;
        *(_DWORD *)(a1 + 64) = v23;
      }
      if ( (unsigned __int8)sub_1643480(v9) )
      {
        v10 = sub_1560BF0(*(__int64 **)v28, &v35);
        v11 = *(_DWORD *)(a2 + 8);
        if ( v11 >= *(_DWORD *)(a2 + 12) )
        {
          v25 = v10;
          sub_38903E0(a2, 0);
          v10 = v25;
          v11 = *(_DWORD *)(a2 + 8);
        }
        v12 = (unsigned __int64 *)(*(_QWORD *)a2 + 56LL * v11);
        if ( v12 )
        {
          v13 = v28;
          v12[2] = v10;
          v12[1] = v13;
          *v12 = v26;
          v14 = v32;
          v12[3] = (unsigned __int64)(v12 + 5);
          sub_3887850((__int64 *)v12 + 3, v14, (__int64)&v14[v33]);
          v11 = *(_DWORD *)(a2 + 8);
        }
        *(_DWORD *)(a2 + 8) = v11 + 1;
        if ( *(_DWORD *)(a1 + 64) == 4 )
        {
          while ( 1 )
          {
            v16 = sub_3887100(v4);
            *(_DWORD *)(a1 + 64) = v16;
            if ( v16 == 2 )
              break;
            v17 = *(_QWORD *)(a1 + 56);
            v31 = 1;
            v26 = v17;
            v29[0] = "expected type";
            v30 = 3;
            if ( (unsigned __int8)sub_3891B00(a1, &v28, (__int64)v29, 0) || (unsigned __int8)sub_388C730(a1, &v35) )
              goto LABEL_4;
            if ( !*(_BYTE *)(v28 + 8) )
              goto LABEL_25;
            if ( *(_DWORD *)(a1 + 64) == 375 )
            {
              sub_2240AE0((unsigned __int64 *)&v32, (unsigned __int64 *)(a1 + 72));
              *(_DWORD *)(a1 + 64) = sub_3887100(v4);
            }
            else
            {
              sub_2241130((unsigned __int64 *)&v32, 0, v33, byte_3F871B3, 0);
            }
            if ( !*(_BYTE *)(v28 + 8) || *(_BYTE *)(v28 + 8) == 12 )
              goto LABEL_23;
            v18 = sub_1560BF0(*(__int64 **)v28, &v35);
            v19 = *(_DWORD *)(a2 + 8);
            if ( v19 >= *(_DWORD *)(a2 + 12) )
            {
              v24 = v18;
              sub_38903E0(a2, 0);
              v18 = v24;
              v19 = *(_DWORD *)(a2 + 8);
            }
            v20 = (unsigned __int64 *)(*(_QWORD *)a2 + 56LL * v19);
            if ( v20 )
            {
              v21 = v28;
              v20[2] = v18;
              v20[1] = v21;
              *v20 = v26;
              v22 = v32;
              v20[3] = (unsigned __int64)(v20 + 5);
              sub_3887850((__int64 *)v20 + 3, v22, (__int64)&v22[v33]);
              v19 = *(_DWORD *)(a2 + 8);
            }
            *(_DWORD *)(a2 + 8) = v19 + 1;
            if ( *(_DWORD *)(a1 + 64) != 4 )
              goto LABEL_20;
          }
          *(_DWORD *)(a1 + 64) = sub_3887100(v4);
          *a3 = 1;
        }
LABEL_20:
        if ( v32 != (_QWORD *)v34 )
          j_j___libc_free_0((unsigned __int64)v32);
        sub_3887AD0(v37);
        return sub_388AF10(a1, 13, "expected ')' at end of argument list");
      }
LABEL_23:
      v31 = 1;
      v15 = "invalid type for function argument";
    }
    else
    {
LABEL_25:
      v31 = 1;
      v15 = "argument can not have void type";
    }
    v29[0] = v15;
    v30 = 3;
    v7 = sub_38814C0(v4, v26, (__int64)v29);
  }
  if ( v32 != (_QWORD *)v34 )
    j_j___libc_free_0((unsigned __int64)v32);
  sub_3887AD0(v37);
  return v7;
}

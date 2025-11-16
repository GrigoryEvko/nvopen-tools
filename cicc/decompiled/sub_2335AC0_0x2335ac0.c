// Function: sub_2335AC0
// Address: 0x2335ac0
//
__int64 __fastcall sub_2335AC0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // al
  __m128i v5; // kr00_16
  char v6; // al
  unsigned int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r15
  unsigned int v10; // edx
  __int64 v11; // rax
  unsigned int v12; // eax
  unsigned int v13; // ebx
  __int64 v14; // rdx
  unsigned int v15; // [rsp+8h] [rbp-D8h]
  __m128i v16; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v17; // [rsp+28h] [rbp-B8h] BYREF
  __m128i v18; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v19; // [rsp+48h] [rbp-98h]
  unsigned __int64 v20[4]; // [rsp+50h] [rbp-90h] BYREF
  const char *v21; // [rsp+70h] [rbp-70h] BYREF
  __int64 v22; // [rsp+78h] [rbp-68h]
  _QWORD *v23; // [rsp+80h] [rbp-60h]
  __int64 v24; // [rsp+88h] [rbp-58h]
  char v25; // [rsp+90h] [rbp-50h]
  void *v26; // [rsp+98h] [rbp-48h] BYREF
  __m128i *v27; // [rsp+A0h] [rbp-40h]
  _QWORD v28[7]; // [rsp+A8h] [rbp-38h] BYREF

  v16.m128i_i64[0] = a2;
  v16.m128i_i64[1] = a3;
  if ( !a3 )
    goto LABEL_2;
  LOBYTE(v21) = 59;
  sub_232E160(&v18, &v16, &v21, 1u);
  if ( v19 )
  {
    v7 = sub_C63BB0();
    v22 = 38;
    v9 = v8;
    v25 = 1;
    v15 = v7;
    v21 = "too many CFGuardPass parameters '{0}' ";
    v23 = v28;
    v24 = 1;
    v27 = &v16;
    v26 = &unk_49DB108;
    v28[0] = &v26;
    sub_23328D0((__int64)v20, (__int64)&v21);
    v10 = v15;
    goto LABEL_9;
  }
  v5 = v18;
  if ( !sub_9691B0((const void *)v18.m128i_i64[0], v18.m128i_u64[1], "check", 5) )
  {
    if ( sub_9691B0((const void *)v5.m128i_i64[0], v5.m128i_u64[1], "dispatch", 8) )
    {
      v6 = *(_BYTE *)(a1 + 8);
      *(_DWORD *)a1 = 1;
      *(_BYTE *)(a1 + 8) = v6 & 0xFC | 2;
      return a1;
    }
    v12 = sub_C63BB0();
    v27 = &v18;
    v13 = v12;
    v9 = v14;
    v21 = "invalid CFGuardPass mechanism: '{0}' ";
    v23 = v28;
    v22 = 37;
    v24 = 1;
    v26 = &unk_49DB108;
    v25 = 1;
    v28[0] = &v26;
    sub_23328D0((__int64)v20, (__int64)&v21);
    v10 = v13;
LABEL_9:
    sub_23058C0(&v17, (__int64)v20, v10, v9);
    v11 = v17;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v11 & 0xFFFFFFFFFFFFFFFELL;
    sub_2240A30(v20);
    return a1;
  }
LABEL_2:
  v3 = *(_BYTE *)(a1 + 8);
  *(_DWORD *)a1 = 0;
  *(_BYTE *)(a1 + 8) = v3 & 0xFC | 2;
  return a1;
}

// Function: sub_69C8B0
// Address: 0x69c8b0
//
__int64 __fastcall sub_69C8B0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  char i; // dl
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  char j; // dl
  __int64 v16; // rdi
  char v18; // [rsp+8h] [rbp-AB8h]
  __int64 v22; // [rsp+28h] [rbp-A98h]
  int v23; // [rsp+38h] [rbp-A88h] BYREF
  _BYTE v24[4]; // [rsp+3Ch] [rbp-A84h] BYREF
  __int64 v25; // [rsp+40h] [rbp-A80h] BYREF
  __int64 v26; // [rsp+48h] [rbp-A78h] BYREF
  _BYTE v27[17]; // [rsp+50h] [rbp-A70h] BYREF
  char v28; // [rsp+61h] [rbp-A5Fh]
  __int64 v29[44]; // [rsp+F0h] [rbp-9D0h] BYREF
  __int64 v30[44]; // [rsp+250h] [rbp-870h] BYREF
  __m128i v31; // [rsp+3B0h] [rbp-710h] BYREF
  char v32; // [rsp+3C0h] [rbp-700h]
  _QWORD v33[44]; // [rsp+510h] [rbp-5B0h] BYREF
  __int64 v34[44]; // [rsp+670h] [rbp-450h] BYREF
  __int64 v35[44]; // [rsp+7D0h] [rbp-2F0h] BYREF
  _QWORD v36[50]; // [rsp+930h] [rbp-190h] BYREF

  v26 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  sub_6E1DD0(&v25);
  sub_6E1E00(5, v27, 0, 1);
  v6 = v26;
  *(_DWORD *)(qword_4D03C50 + 18LL) |= 0x10080u;
  sub_68B920(*a1, v6, (__int64)v29, (__int64)v30);
  sub_68FEF0(v29, v30, &dword_4F063F8, dword_4F06650[0], 0, (__int64)&v31);
  if ( !v32 )
    goto LABEL_13;
  v7 = v31.m128i_i64[0];
  for ( i = *(_BYTE *)(v31.m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v7 + 140) )
    v7 = *(_QWORD *)(v7 + 160);
  if ( !i )
  {
LABEL_13:
    sub_6E4710(&v31);
    sub_724E30(&v26);
    sub_6E2B30(&v26, v30);
    sub_6E1DF0(v25);
    sub_6E1DD0(&v25);
    sub_6E1E00(5, v27, 0, 0);
    v28 |= 3u;
    sub_6E7150(a1, v29);
    sub_6E7150(a2, v30);
    goto LABEL_14;
  }
  v18 = *(_BYTE *)(qword_4D03C50 + 19LL) & 1;
  sub_6E4710(&v31);
  sub_724E30(&v26);
  sub_6E2B30(&v26, v30);
  sub_6E1DF0(v25);
  sub_6E1DD0(&v25);
  sub_6E1E00(5, v27, 0, 0);
  v28 |= 3u;
  sub_6E7150(a1, v29);
  sub_6E7150(a2, v30);
  if ( v18 )
  {
LABEL_14:
    v23 = 0;
    sub_68B310(a3, &v23);
    if ( (unsigned int)(v23 - 1) <= 1 || v23 == 4 )
    {
      sub_72CF70();
      sub_6FF9F0(v29, v34, 1, v24, 1);
      sub_6FF9F0(v30, v35, 1, v24, 1);
      sub_69B310(v34, v35, 0x2Bu, dword_4F07508, dword_4F06650[0], (__int64)v33);
      sub_688FA0(v33);
      switch ( v23 )
      {
        case 2:
          sub_6E6A50(unk_4F06C10, v34);
          sub_6E6A50(unk_4F06C08, v35);
          break;
        case 4:
          sub_6FF9F0(v29, v34, 1, v24, 1);
          sub_6FF9F0(v30, v35, 1, v24, 1);
          sub_69B310(v35, v34, 0x2Bu, dword_4F07508, dword_4F06650[0], (__int64)v36);
          sub_688FA0(v36);
          sub_6E6A50(unk_4F06BF0, v34);
          sub_6E6A50(unk_4F06BE8, &v31);
          sub_6F7CC0((unsigned int)v36, (unsigned int)v34, (unsigned int)&v31, a3, 0, 0, (__int64)v35);
          sub_6E6A50(unk_4F06BF8, v34);
          break;
        case 1:
          sub_6E6A50(unk_4F06C28, v34);
          sub_6E6A50(unk_4F06C20, v35);
          break;
        default:
LABEL_21:
          sub_721090(v33);
      }
      sub_6F7CC0((unsigned int)v33, (unsigned int)v34, (unsigned int)v35, a3, 0, 0, (__int64)v36);
    }
    else
    {
      sub_72CF70();
    }
    sub_6907F0(v29, v30, 0x2Fu, dword_4F07508, dword_4F06650[0], (__int64)v33);
    sub_688FA0(v33);
    switch ( v23 )
    {
      case 1:
        sub_6E6A50(unk_4F06C30, v29);
        goto LABEL_27;
      case 2:
        sub_6E6A50(unk_4F06C18, v29);
        goto LABEL_27;
      case 4:
        sub_6E6A50(unk_4F06C00, v29);
        goto LABEL_27;
      case 8:
        sub_6E6A50(unk_4F06BE0, v29);
        sub_6E6A50(unk_4F06BD8, v36);
        goto LABEL_27;
      case 16:
        sub_6E6A50(unk_4F06BD0, v29);
        sub_6E6A50(unk_4F06BC8, v36);
LABEL_27:
        sub_6F7CC0((unsigned int)v33, (unsigned int)v29, (unsigned int)v36, a3, 0, 0, (__int64)&v31);
        goto LABEL_7;
      default:
        goto LABEL_21;
    }
  }
  sub_68FEF0(v29, v30, dword_4F07508, dword_4F06650[0], 0, (__int64)&v31);
LABEL_7:
  v9 = sub_736020(a3, 0);
  sub_68BC10(v9, &v31);
  sub_6E2B30(v9, &v31);
  sub_6E1E00(5, v27, 0, 0);
  sub_6F8E70(v9, dword_4F07508, dword_4F07508, v29, 0);
  v22 = sub_72BA30(5);
  v36[0] = sub_724DC0(5, dword_4F07508, v10, v11, v12, v13);
  sub_72BB40(v22, v36[0]);
  sub_6E6A50(v36[0], v30);
  sub_724E30(v36);
  sub_6907F0(v29, v30, 0x30u, dword_4F07508, dword_4F06650[0], (__int64)&v31);
  if ( v32 )
  {
    v14 = v31.m128i_i64[0];
    for ( j = *(_BYTE *)(v31.m128i_i64[0] + 140); j == 12; j = *(_BYTE *)(v14 + 140) )
      v14 = *(_QWORD *)(v14 + 160);
    if ( j )
      sub_688FA0(&v31);
  }
  v16 = sub_6F6F40(&v31, 0);
  *a4 = v16;
  *a4 = sub_6E2700(v16);
  sub_6E2B30(v16, 0);
  sub_6E1DF0(v25);
  return v9;
}

// Function: sub_701AC0
// Address: 0x701ac0
//
__int64 __fastcall sub_701AC0(__int64 a1, int a2, __int64 a3, __int64 a4, _DWORD *a5)
{
  __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // rdx
  int v11; // r8d
  char v12; // cl
  __m128i *v13; // rsi
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdi
  __int64 v26; // r13
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // [rsp+8h] [rbp-258h]
  __int64 v31; // [rsp+10h] [rbp-250h]
  __m128i v32; // [rsp+18h] [rbp-248h] BYREF
  __int64 v33; // [rsp+28h] [rbp-238h]
  _BYTE v34[160]; // [rsp+30h] [rbp-230h] BYREF
  __m128i v35[25]; // [rsp+D0h] [rbp-190h] BYREF

  v32.m128i_i64[0] = a1;
  v8 = qword_4D03C50;
  qword_4D03C50 = 0;
  v31 = v8;
  sub_6E1E00(4u, (__int64)v34, 0, 1);
  if ( *(_BYTE *)(a4 + 24) == 10 )
  {
    v30 = *(_QWORD *)(a4 + 64);
    sub_733650(v30);
    v28 = qword_4F06BC0;
    qword_4F06BC0 = v30;
    v29 = qword_4D03C50;
    *(_QWORD *)(v30 + 32) = v28;
    *(_QWORD *)(v29 + 48) = v30;
    a4 = *(_QWORD *)(a4 + 56);
  }
  else
  {
    sub_733780(0, 0, 0, 4, 0);
    *(_QWORD *)(qword_4D03C50 + 48LL) = qword_4F06BC0;
  }
  v9 = (_QWORD *)sub_7312D0(v32.m128i_i64[0]);
  v10 = v32.m128i_i64[0];
  v9[2] = a3;
  *(_QWORD *)(a3 + 16) = a4;
  if ( a2 && (*(_BYTE *)(v10 + 192) & 2) != 0 )
  {
    v12 = 0;
    v11 = 1;
  }
  else
  {
    v11 = 0;
    v12 = a2 == 0 && (*(_BYTE *)(v10 + 192) & 2) != 0;
  }
  v13 = *(__m128i **)(v10 + 152);
  v16 = (_QWORD *)sub_6FD870(v9, (__int64)v13, v10, v12, v11, 1, 1, 0, 0, 0, 0, a5, 0);
  if ( (*(_BYTE *)(v32.m128i_i64[0] + 193) & 4) != 0
    && (!qword_4F04C50 || (*(_BYTE *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 193LL) & 4) == 0) )
  {
    v32.m128i_i64[1] = 0;
    v13 = &v32;
    v33 = 0;
    if ( (unsigned int)sub_7016D0((__int64)v16, &v32, v35, (__m128i *)&v32.m128i_u64[1], v14, v15) )
    {
      v13 = 0;
      v16 = (_QWORD *)sub_6F6F40(v35, 0, v17, v18, v19, v20);
    }
    sub_67E3D0(&v32.m128i_i64[1]);
  }
  if ( (unsigned int)sub_8D32E0(*v16) )
    v16 = (_QWORD *)sub_73DDB0(v16);
  v25 = sub_6E2700((__int64)v16, v13, v21, v22, v23, v24);
  v26 = sub_732B10(v25);
  sub_6E2B30(v25, (__int64)v13);
  qword_4D03C50 = v31;
  return v26;
}

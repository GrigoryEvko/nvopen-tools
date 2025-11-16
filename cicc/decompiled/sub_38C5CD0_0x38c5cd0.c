// Function: sub_38C5CD0
// Address: 0x38c5cd0
//
__int64 __fastcall sub_38C5CD0(__int64 *a1, __int64 *a2, int a3, char *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  void (__fastcall *v16)(__int64 *, __int64, __int64); // rcx
  unsigned int v17; // r15d
  __int64 v18; // r11
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // rax
  char *v24; // r15
  char *v25; // rbx
  __int64 v26; // rsi
  __int64 v28; // rdx
  __int64 v30; // [rsp+8h] [rbp-68h]
  __int64 v32; // [rsp+20h] [rbp-50h]
  __int64 v33; // [rsp+28h] [rbp-48h]
  unsigned int v34; // [rsp+34h] [rbp-3Ch]

  v8 = *a1;
  v9 = a2[1];
  if ( !*a1 )
    v8 = sub_38BFA60(v9, 1);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a2 + 176))(a2, v8, 0);
  v33 = sub_38BFA60(v9, 1);
  v10 = sub_38CF310(v33, 0, a2[1], 0);
  v11 = sub_38CF310(v8, 0, a2[1], 0);
  v12 = sub_38CB1F0(17, v10, v11, a2[1], 0);
  v13 = sub_38CB470(4, a2[1]);
  v14 = sub_38CB1F0(17, v12, v13, a2[1], 0);
  sub_38C4F40(a2, v14, 4u);
  v15 = *(unsigned int *)(*(_QWORD *)(v9 + 32) + 740LL);
  if ( (unsigned int)v15 <= 0x1E )
  {
    v16 = *(void (__fastcall **)(__int64 *, __int64, __int64))(*a2 + 424);
    v28 = 1610614920;
    if ( _bittest64(&v28, v15) )
    {
      v16(a2, 2, 2);
      v34 = 2;
      v18 = 10;
      goto LABEL_7;
    }
  }
  else
  {
    v16 = *(void (__fastcall **)(__int64 *, __int64, __int64))(*a2 + 424);
  }
  v17 = *(unsigned __int16 *)(v9 + 1160);
  v34 = v17;
  v16(a2, *(unsigned __int16 *)(v9 + 1160), 2);
  if ( v17 <= 4 )
  {
    v18 = 10;
  }
  else
  {
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a2 + 424))(
      a2,
      *(unsigned int *)(*(_QWORD *)(v9 + 16) + 8LL),
      1);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a2 + 424))(a2, 0, 1);
    v18 = 12;
  }
LABEL_7:
  v30 = v18;
  v32 = sub_38BFA60(v9, 1);
  v19 = sub_38CF310(v32, 0, a2[1], 0);
  v20 = sub_38CF310(v8, 0, a2[1], 0);
  v21 = sub_38CB1F0(17, v19, v20, a2[1], 0);
  v22 = sub_38CB470(v30, a2[1]);
  v23 = sub_38CB1F0(17, v21, v22, a2[1], 0);
  sub_38C4F40(a2, v23, 4u);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a2 + 424))(
    a2,
    *(unsigned int *)(*(_QWORD *)(v9 + 16) + 28LL),
    1);
  if ( v34 > 3 )
    (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a2 + 424))(a2, 1, 1);
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a2 + 424))(a2, 1, 1);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a2 + 424))(a2, SBYTE1(a3), 1);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a2 + 424))(a2, BYTE2(a3), 1);
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a2 + 424))(a2, a5 + 1, 1);
  v24 = a4;
  v25 = &a4[a5];
  if ( a4 != v25 )
  {
    do
    {
      v26 = *v24++;
      (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a2 + 424))(a2, v26, 1);
    }
    while ( v25 != v24 );
  }
  if ( v34 <= 4 )
    sub_38C5870((__int64)a1, (__int64)a2);
  else
    sub_38C59C0((__int64)a1, a2, a6, *(_QWORD **)(v9 + 752), *(unsigned int *)(v9 + 760));
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a2 + 176))(a2, v32, 0);
  return v8;
}

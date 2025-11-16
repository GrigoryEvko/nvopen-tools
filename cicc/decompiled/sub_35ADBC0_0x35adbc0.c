// Function: sub_35ADBC0
// Address: 0x35adbc0
//
void __fastcall sub_35ADBC0(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // r13
  __int64 *v5; // rdi
  __int64 v6; // rax
  __int64 (*v7)(void); // rdx
  __int64 (*v8)(); // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // r13
  _QWORD *v11; // r11
  __int64 *v12; // r14
  __int64 v13; // rax
  __int64 (*v14)(); // rax
  __int64 v15; // rax
  _QWORD *v16; // rsi
  __int32 v17; // eax
  __int64 v18; // rdx
  unsigned __int32 v19; // ecx
  __int64 *v20; // rax
  unsigned int v21; // [rsp+Ch] [rbp-A4h]
  __int64 v22; // [rsp+10h] [rbp-A0h]
  _QWORD *v23; // [rsp+18h] [rbp-98h]
  __int64 v24; // [rsp+18h] [rbp-98h]
  __int64 v25; // [rsp+28h] [rbp-88h] BYREF
  __int64 v26[4]; // [rsp+30h] [rbp-80h] BYREF
  __m128i v27; // [rsp+50h] [rbp-60h] BYREF
  __int64 v28; // [rsp+60h] [rbp-50h]
  __int64 v29; // [rsp+68h] [rbp-48h]
  __int64 v30; // [rsp+70h] [rbp-40h]

  v3 = 0;
  v4 = *(_QWORD *)(a1 + 32);
  v5 = *(__int64 **)(v4 + 16);
  v6 = *v5;
  v7 = *(__int64 (**)(void))(*v5 + 128);
  if ( v7 != sub_2DAC790 )
  {
    v3 = (_QWORD *)v7();
    v6 = **(_QWORD **)(v4 + 16);
  }
  v8 = *(__int64 (**)())(v6 + 136);
  if ( v8 == sub_2DD19D0 )
  {
    (*(void (**)(void))(v6 + 200))();
    sub_2E313E0(a1);
    BUG();
  }
  v23 = (_QWORD *)v8();
  v22 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v4 + 16) + 200LL))(*(_QWORD *)(v4 + 16));
  v9 = sub_2E313E0(a1);
  v10 = a2[1];
  v11 = v23;
  v12 = (__int64 *)v9;
  v13 = *v23;
  v24 = *a2;
  v14 = *(__int64 (**)())(v13 + 192);
  if ( v14 != sub_2FDBC40 )
  {
    if ( ((unsigned __int8 (__fastcall *)(_QWORD *, __int64, __int64 *, _QWORD, unsigned __int64, __int64))v14)(
           v11,
           a1,
           v12,
           *a2,
           0xAAAAAAAAAAAAAAABLL * ((__int64)(a2[1] - *a2) >> 2),
           v22) )
    {
      return;
    }
    v10 = a2[1];
    v24 = *a2;
  }
  for ( ; v24 != v10; v10 -= 12 )
  {
    while ( 1 )
    {
      v19 = *(_DWORD *)(v10 - 12);
      if ( *(_BYTE *)(v10 - 3) )
        break;
      v21 = *(_DWORD *)(v10 - 12);
      v10 -= 12;
      v20 = sub_2FF6500(v22, v19, 1);
      (*(void (__fastcall **)(_QWORD *, __int64, __int64 *, _QWORD, _QWORD, __int64 *, __int64, _QWORD, _QWORD))(*v3 + 568LL))(
        v3,
        a1,
        v12,
        v21,
        *(unsigned int *)(v10 + 4),
        v20,
        v22,
        0,
        0);
      if ( v24 == v10 )
        return;
    }
    v15 = v3[1];
    memset(v26, 0, 24);
    v25 = 0;
    v16 = sub_2F26260(a1, v12, v26, v15 - 800, v19);
    v17 = *(_DWORD *)(v10 - 8);
    v27.m128i_i64[0] = 0x40000000;
    v28 = 0;
    v27.m128i_i32[2] = v17;
    v29 = 0;
    v30 = 0;
    sub_2E8EAD0(v18, (__int64)v16, &v27);
    if ( v26[0] )
      sub_B91220((__int64)v26, v26[0]);
    if ( v25 )
      sub_B91220((__int64)&v25, v25);
  }
}

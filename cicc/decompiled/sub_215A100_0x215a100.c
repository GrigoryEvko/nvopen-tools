// Function: sub_215A100
// Address: 0x215a100
//
void __fastcall sub_215A100(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rdi
  __int64 v5; // r13
  __int64 v6; // r14
  _BYTE *v7; // rsi
  _BYTE *v8; // rax
  __int64 v9; // rdi
  unsigned __int64 v10; // rdx
  _QWORD v11[2]; // [rsp+20h] [rbp-1B0h] BYREF
  _QWORD *v12; // [rsp+30h] [rbp-1A0h] BYREF
  __int16 v13; // [rsp+40h] [rbp-190h]
  __int64 v14; // [rsp+50h] [rbp-180h] BYREF
  __int64 v15; // [rsp+58h] [rbp-178h]
  __int64 v16; // [rsp+60h] [rbp-170h]
  __int64 v17; // [rsp+68h] [rbp-168h]
  __int64 v18; // [rsp+70h] [rbp-160h] BYREF
  __int64 v19; // [rsp+78h] [rbp-158h]
  __int64 v20; // [rsp+80h] [rbp-150h]
  __int64 v21; // [rsp+88h] [rbp-148h]
  _QWORD v22[2]; // [rsp+90h] [rbp-140h] BYREF
  unsigned __int64 v23; // [rsp+A0h] [rbp-130h]
  _BYTE *v24; // [rsp+A8h] [rbp-128h]
  int v25; // [rsp+B0h] [rbp-120h]
  unsigned __int64 *v26; // [rsp+B8h] [rbp-118h]
  _BYTE *v27; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v28; // [rsp+C8h] [rbp-108h]
  _BYTE v29[64]; // [rsp+D0h] [rbp-100h] BYREF
  unsigned __int64 v30[2]; // [rsp+110h] [rbp-C0h] BYREF
  _BYTE v31[176]; // [rsp+120h] [rbp-B0h] BYREF

  v30[0] = (unsigned __int64)v31;
  v30[1] = 0x8000000000LL;
  v22[0] = &unk_49EFC48;
  v26 = v30;
  v25 = 1;
  v24 = 0;
  v23 = 0;
  v22[1] = 0;
  sub_16E7A40((__int64)v22, 0, 0, 0);
  sub_2159D00(a1, a2, (__int64)v22);
  v14 = 0;
  v27 = v29;
  v28 = 0x800000000LL;
  v3 = *(_QWORD *)(a2 + 16);
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  if ( v3 != a2 + 8 )
  {
    do
    {
      v4 = v3 - 56;
      if ( !v3 )
        v4 = 0;
      sub_2157D50(v4, (__int64)&v27, (__int64)&v14, (__int64)&v18);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( a2 + 8 != v3 );
    if ( (_DWORD)v28 )
    {
      v5 = 8LL * (unsigned int)v28;
      v6 = 0;
      do
      {
        v7 = *(_BYTE **)&v27[v6];
        v6 += 8;
        sub_2156420(a1, v7, (__int64)v22, 0);
      }
      while ( v5 != v6 );
    }
  }
  v8 = v24;
  if ( (unsigned __int64)v24 >= v23 )
  {
    sub_16E7DE0((__int64)v22, 10);
  }
  else
  {
    ++v24;
    *v8 = 10;
  }
  v9 = *(_QWORD *)(a1 + 256);
  v10 = *v26;
  v11[1] = *((unsigned int *)v26 + 2);
  v13 = 261;
  v11[0] = v10;
  v12 = v11;
  sub_38DD5A0(v9, &v12);
  j___libc_free_0(v19);
  j___libc_free_0(v15);
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  v22[0] = &unk_49EFD28;
  sub_16E7960((__int64)v22);
  if ( (_BYTE *)v30[0] != v31 )
    _libc_free(v30[0]);
}

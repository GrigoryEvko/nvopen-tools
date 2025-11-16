// Function: sub_302CFC0
// Address: 0x302cfc0
//
void __fastcall sub_302CFC0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rdi
  _BYTE **v5; // r14
  __int64 v6; // r15
  _BYTE **v7; // r13
  _BYTE *v8; // rsi
  _BYTE *v9; // rax
  __int64 *v10; // rdi
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // [rsp+20h] [rbp-1D0h] BYREF
  __int64 v14; // [rsp+28h] [rbp-1C8h]
  __int64 v15; // [rsp+30h] [rbp-1C0h]
  __int64 v16; // [rsp+38h] [rbp-1B8h]
  __int64 v17; // [rsp+40h] [rbp-1B0h] BYREF
  __int64 v18; // [rsp+48h] [rbp-1A8h]
  __int64 v19; // [rsp+50h] [rbp-1A0h]
  __int64 v20; // [rsp+58h] [rbp-198h]
  _QWORD v21[4]; // [rsp+60h] [rbp-190h] BYREF
  __int16 v22; // [rsp+80h] [rbp-170h]
  _QWORD v23[3]; // [rsp+90h] [rbp-160h] BYREF
  unsigned __int64 v24; // [rsp+A8h] [rbp-148h]
  _BYTE *v25; // [rsp+B0h] [rbp-140h]
  __int64 v26; // [rsp+B8h] [rbp-138h]
  unsigned __int64 *v27; // [rsp+C0h] [rbp-130h]
  _BYTE *v28; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v29; // [rsp+D8h] [rbp-118h]
  _BYTE v30[64]; // [rsp+E0h] [rbp-110h] BYREF
  unsigned __int64 v31[3]; // [rsp+120h] [rbp-D0h] BYREF
  _BYTE v32[184]; // [rsp+138h] [rbp-B8h] BYREF

  v31[0] = (unsigned __int64)v32;
  v26 = 0x100000000LL;
  v23[0] = &unk_49DD288;
  v27 = v31;
  v31[1] = 0;
  v31[2] = 128;
  v23[1] = 2;
  v23[2] = 0;
  v24 = 0;
  v25 = 0;
  sub_CB5980((__int64)v23, 0, 0, 0);
  sub_3026120(a1, a2, (__int64)v23);
  v13 = 0;
  v28 = v30;
  v29 = 0x800000000LL;
  v3 = *(_QWORD *)(a2 + 16);
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  if ( v3 != a2 + 8 )
  {
    do
    {
      v4 = v3 - 56;
      if ( !v3 )
        v4 = 0;
      sub_302CB00(v4, (__int64)&v28, (__int64)&v13, (__int64)&v17);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( a2 + 8 != v3 );
    v5 = (_BYTE **)v28;
    v6 = *(_QWORD *)(a1 + 200) + 1288LL;
    v7 = (_BYTE **)&v28[8 * (unsigned int)v29];
    if ( v7 != (_BYTE **)v28 )
    {
      do
      {
        v8 = *v5++;
        sub_3029BF0(a1, v8, (__int64)v23, 0, v6);
      }
      while ( v7 != v5 );
    }
  }
  v9 = v25;
  if ( (unsigned __int64)v25 >= v24 )
  {
    sub_CB5D20((__int64)v23, 10);
  }
  else
  {
    ++v25;
    *v9 = 10;
  }
  v10 = *(__int64 **)(a1 + 224);
  v11 = v27[1];
  v12 = *v27;
  v22 = 261;
  v21[0] = v12;
  v21[1] = v11;
  sub_E99A90(v10, (__int64)v21);
  sub_C7D6A0(v18, 8LL * (unsigned int)v20, 8);
  sub_C7D6A0(v14, 8LL * (unsigned int)v16, 8);
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
  v23[0] = &unk_49DD388;
  sub_CB5840((__int64)v23);
  if ( (_BYTE *)v31[0] != v32 )
    _libc_free(v31[0]);
}

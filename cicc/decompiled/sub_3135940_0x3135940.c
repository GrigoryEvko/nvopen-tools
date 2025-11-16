// Function: sub_3135940
// Address: 0x3135940
//
void __fastcall sub_3135940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 *v7; // rdx
  __int64 *v8; // rax
  __int64 *i; // rdx
  __int64 v10; // rbx
  unsigned __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // ebx
  __int64 *v16; // rax
  __int64 **v17; // rbx
  _QWORD *v18; // r12
  __int64 v19; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v20; // [rsp+8h] [rbp-D8h]
  __int64 v22; // [rsp+28h] [rbp-B8h]
  _QWORD v23[4]; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v24; // [rsp+50h] [rbp-90h]
  __int64 *v25; // [rsp+60h] [rbp-80h] BYREF
  __int64 v26; // [rsp+68h] [rbp-78h]
  _BYTE v27[112]; // [rsp+70h] [rbp-70h] BYREF

  if ( a5 )
  {
    v25 = (__int64 *)v27;
    v7 = (__int64 *)v27;
    v26 = 0x800000000LL;
    v8 = (__int64 *)v27;
    if ( a5 > 8 )
    {
      v20 = a5;
      sub_C8D5F0((__int64)&v25, v27, a5, 8u, a5, a6);
      v7 = v25;
      a5 = v20;
      v8 = &v25[(unsigned int)v26];
    }
    for ( i = &v7[a5]; i != v8; ++v8 )
    {
      if ( v8 )
        *v8 = 0;
    }
    LODWORD(v26) = a5;
    if ( (_DWORD)a5 )
    {
      v10 = a4 + 16;
      v11 = 0;
      v12 = 8LL * (unsigned int)a5;
      do
      {
        v10 += 24;
        v13 = sub_BCE3C0(*(__int64 **)(a1 + 584), 0);
        v14 = sub_ADB060(*(_QWORD *)(v10 - 24), v13);
        v25[v11 / 8] = v14;
        v11 += 8LL;
      }
      while ( v11 != v12 );
      v15 = v26;
      if ( (_DWORD)v26 )
      {
        v16 = (__int64 *)sub_BCE3C0(*(__int64 **)(a1 + 584), 0);
        v17 = (__int64 **)sub_BCD420(v16, v15);
        BYTE4(v22) = 0;
        v19 = sub_AD1300(v17, v25, (unsigned int)v26);
        v24 = 261;
        v23[0] = a2;
        v23[1] = a3;
        v18 = sub_BD2C40(88, unk_3F0FAE8);
        if ( v18 )
          sub_B30000((__int64)v18, *(_QWORD *)(a1 + 504), v17, 0, 6, v19, (__int64)v23, 0, 0, v22, 0);
        sub_B31A00((__int64)v18, (__int64)"llvm.metadata", 13);
      }
    }
    if ( v25 != (__int64 *)v27 )
      _libc_free((unsigned __int64)v25);
  }
}

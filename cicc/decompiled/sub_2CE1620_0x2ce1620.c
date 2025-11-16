// Function: sub_2CE1620
// Address: 0x2ce1620
//
void __fastcall sub_2CE1620(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 i; // r14
  unsigned __int64 v8; // r15
  int v9; // edx
  __int64 v10; // rdx
  _QWORD *v11; // r12
  __int64 v12; // r8
  _QWORD *v13; // rax
  int v14; // ecx
  _BYTE *v15; // rdx
  __int64 v16; // rsi
  unsigned __int64 v17; // r12
  unsigned __int64 *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // rcx
  char v22; // di
  unsigned __int64 v24; // [rsp+8h] [rbp-98h]
  int v25; // [rsp+8h] [rbp-98h]
  _BYTE *v28; // [rsp+20h] [rbp-80h] BYREF
  __int64 v29; // [rsp+28h] [rbp-78h]
  _BYTE v30[112]; // [rsp+30h] [rbp-70h] BYREF

  for ( i = *(_QWORD *)(a3 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v8 = *(_QWORD *)(i + 24);
    if ( *(_BYTE *)v8 <= 0x1Cu )
      BUG();
    if ( *(_BYTE *)v8 == 63 )
    {
      v9 = *(_DWORD *)(v8 + 4);
      v28 = v30;
      v10 = v9 & 0x7FFFFFF;
      v29 = 0x800000000LL;
      v11 = (_QWORD *)(v8 + 32 * (1 - v10));
      v12 = (-32 * (1 - v10)) >> 5;
      if ( (unsigned __int64)(-32 * (1 - v10)) > 0x100 )
      {
        v25 = (-32 * (1 - v10)) >> 5;
        sub_C8D5F0((__int64)&v28, v30, (-32 * (1 - v10)) >> 5, 8u, v12, a6);
        v15 = v28;
        v14 = v29;
        LODWORD(v12) = v25;
        v13 = &v28[8 * (unsigned int)v29];
      }
      else
      {
        v13 = v30;
        v14 = 0;
        v15 = v30;
      }
      if ( v11 != (_QWORD *)v8 )
      {
        do
        {
          if ( v13 )
            *v13 = *v11;
          v11 += 4;
          ++v13;
        }
        while ( (_QWORD *)v8 != v11 );
        v15 = v28;
        v14 = v29;
      }
      v16 = *(_QWORD *)(v8 + 72);
      LODWORD(v29) = v14 + v12;
      v17 = sub_AE54E0(*(_QWORD *)(a1 + 352), v16, v15, (unsigned int)(v14 + v12)) + a4;
      v18 = (unsigned __int64 *)sub_22077B0(0x38u);
      v18[4] = v8;
      v18[6] = v17;
      v18[5] = a2;
      v24 = (unsigned __int64)v18;
      v19 = sub_2CE1580(a5, v18 + 4);
      if ( v20 )
      {
        v21 = (_QWORD *)(a5 + 8);
        v22 = 1;
        if ( !v19 && v20 != v21 )
          v22 = v8 < v20[4];
        sub_220F040(v22, v24, v20, v21);
        ++*(_QWORD *)(a5 + 40);
      }
      else
      {
        j_j___libc_free_0(v24);
      }
      sub_2CE1620(a1, a2, v8, v17, a5);
      if ( v28 != v30 )
        _libc_free((unsigned __int64)v28);
    }
  }
}

// Function: sub_25DD070
// Address: 0x25dd070
//
void __fastcall sub_25DD070(__int64 a1, __int64 a2)
{
  unsigned int v3; // r14d
  __int64 *v4; // rax
  __int64 v5; // rax
  bool v6; // zf
  __int64 *v7; // rbx
  unsigned __int64 *v8; // rax
  __int64 v9; // rdx
  unsigned __int64 *v10; // r14
  unsigned __int64 v11; // rdi
  unsigned __int64 *v12; // r15
  size_t v13; // rsi
  __int64 **v14; // r14
  __int64 v15; // r15
  unsigned __int8 *v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rdx
  unsigned __int64 v22; // r9
  unsigned __int64 *v23; // rax
  __int64 v24; // [rsp+0h] [rbp-D0h]
  __int64 v25; // [rsp+0h] [rbp-D0h]
  __int64 v26; // [rsp+18h] [rbp-B8h]
  _BYTE v27[32]; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v28; // [rsp+40h] [rbp-90h]
  void *base; // [rsp+50h] [rbp-80h] BYREF
  __int64 v30; // [rsp+58h] [rbp-78h]
  _BYTE v31[112]; // [rsp+60h] [rbp-70h] BYREF

  if ( *(_DWORD *)(a2 + 24) == *(_DWORD *)(a2 + 20) )
  {
    sub_B30290(a1);
  }
  else
  {
    v3 = *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL);
    v4 = (__int64 *)sub_BD5C60(a1);
    v5 = sub_BCE3C0(v4, v3 >> 8);
    v6 = *(_BYTE *)(a2 + 28) == 0;
    v7 = (__int64 *)v5;
    base = v31;
    v30 = 0x800000000LL;
    v8 = *(unsigned __int64 **)(a2 + 8);
    if ( v6 )
      v9 = *(unsigned int *)(a2 + 16);
    else
      v9 = *(unsigned int *)(a2 + 20);
    v10 = &v8[v9];
    if ( v8 == v10 )
      goto LABEL_7;
    while ( 1 )
    {
      v11 = *v8;
      v12 = v8;
      if ( *v8 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v8 )
        goto LABEL_7;
    }
    if ( v10 == v8 )
    {
LABEL_7:
      v13 = 0;
    }
    else
    {
      do
      {
        v19 = sub_ADB060(v11, (__int64)v7);
        v21 = (unsigned int)v30;
        v22 = (unsigned int)v30 + 1LL;
        if ( v22 > HIDWORD(v30) )
        {
          v25 = v19;
          sub_C8D5F0((__int64)&base, v31, (unsigned int)v30 + 1LL, 8u, v20, v22);
          v21 = (unsigned int)v30;
          v19 = v25;
        }
        *((_QWORD *)base + v21) = v19;
        v13 = (unsigned int)(v30 + 1);
        v23 = v12 + 1;
        LODWORD(v30) = v30 + 1;
        if ( v12 + 1 == v10 )
          break;
        while ( 1 )
        {
          v11 = *v23;
          v12 = v23;
          if ( *v23 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v10 == ++v23 )
            goto LABEL_20;
        }
      }
      while ( v10 != v23 );
LABEL_20:
      if ( 8 * v13 > 8 )
      {
        qsort(base, v13, 8u, (__compar_fn_t)sub_25DC5D0);
        v13 = (unsigned int)v30;
      }
    }
    v14 = (__int64 **)sub_BCD420(v7, v13);
    v24 = *(_QWORD *)(a1 + 40);
    sub_B30110((_QWORD *)a1);
    BYTE4(v26) = 0;
    v15 = sub_AD1300(v14, (__int64 *)base, (unsigned int)v30);
    v28 = 257;
    v16 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
    if ( v16 )
      sub_B30000((__int64)v16, v24, v14, 0, 6, v15, (__int64)v27, 0, 0, v26, 0);
    sub_BD6B90(v16, (unsigned __int8 *)a1);
    sub_B31A00((__int64)v16, (__int64)"llvm.metadata", 13);
    sub_B30220(a1);
    *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0xF8000000 | 1;
    sub_B2F9E0(a1, (__int64)"llvm.metadata", v17, v18);
    sub_BD2DD0(a1);
    if ( base != v31 )
      _libc_free((unsigned __int64)base);
  }
}

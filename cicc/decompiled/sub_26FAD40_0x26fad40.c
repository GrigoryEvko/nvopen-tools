// Function: sub_26FAD40
// Address: 0x26fad40
//
void __fastcall sub_26FAD40(
        __int64 a1,
        char a2,
        __int64 a3,
        char a4,
        __int64 (__fastcall *a5)(__int64, __int64, __int64),
        __int64 a6)
{
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  int v11; // ecx
  __int64 v12; // rsi
  int v13; // ecx
  unsigned int v14; // eax
  __int64 v15; // rdi
  int v16; // r9d
  __int64 *v17; // rsi
  __int64 *v18; // rcx
  __int64 v19; // rdx
  unsigned __int8 v20; // al
  __int64 v21; // rdx
  _DWORD *v22; // rax
  unsigned __int64 v23; // rdx
  char v24; // r14
  __int64 v27; // [rsp+20h] [rbp-60h]
  __int64 *v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+38h] [rbp-48h]
  _QWORD v31[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( (unsigned __int8)sub_26FAD20(a2) )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = a1 + 8;
    while ( v8 != v7 )
    {
      while ( 1 )
      {
        if ( !v7 )
          BUG();
        if ( (*(_BYTE *)(v7 - 49) & 0x20) == 0 || !sub_B91C10(v7 - 56, 19) || (unsigned int)sub_B92110(v7 - 56) )
          goto LABEL_4;
        sub_B2F930(&v29, v7 - 56);
        v9 = sub_B2F650((__int64)v29, v30);
        v10 = v9;
        if ( v29 != v31 )
        {
          v27 = v9;
          j_j___libc_free_0((unsigned __int64)v29);
          v10 = v27;
        }
        v11 = *(_DWORD *)(a3 + 24);
        v12 = *(_QWORD *)(a3 + 8);
        if ( v11 )
        {
          v13 = v11 - 1;
          v14 = v13 & (((0xBF58476D1CE4E5B9LL * v10) >> 31) ^ (484763065 * v10));
          v15 = *(_QWORD *)(v12 + 8LL * v14);
          if ( v10 == v15 )
            goto LABEL_4;
          v16 = 1;
          while ( v15 != -1 )
          {
            v14 = v13 & (v16 + v14);
            v15 = *(_QWORD *)(v12 + 8LL * v14);
            if ( v15 == v10 )
              goto LABEL_4;
            ++v16;
          }
        }
        if ( a4 )
          break;
LABEL_24:
        sub_B9D920(v7 - 56, 1u);
        v7 = *(_QWORD *)(v7 + 8);
        if ( v8 == v7 )
          return;
      }
      v29 = v31;
      v30 = 0x200000000LL;
      sub_B91D10(v7 - 56, 19, (__int64)&v29);
      v17 = &v29[(unsigned int)v30];
      if ( v29 == v17 )
      {
LABEL_28:
        if ( v29 != v31 )
          _libc_free((unsigned __int64)v29);
        goto LABEL_24;
      }
      v18 = v29;
      while ( 1 )
      {
        v19 = *v18;
        v20 = *(_BYTE *)(*v18 - 16);
        v21 = (v20 & 2) != 0 ? *(_QWORD *)(v19 - 32) : -16 - 8LL * ((v20 >> 2) & 0xF) + v19;
        if ( !**(_BYTE **)(v21 + 8) )
          break;
        if ( v17 == ++v18 )
          goto LABEL_28;
      }
      v22 = (_DWORD *)sub_B91420(*(_QWORD *)(v21 + 8));
      v24 = sub_26F9C90(v22, v23, a5, a6);
      if ( v29 != v31 )
        _libc_free((unsigned __int64)v29);
      if ( !v24 )
        goto LABEL_24;
LABEL_4:
      v7 = *(_QWORD *)(v7 + 8);
    }
  }
}

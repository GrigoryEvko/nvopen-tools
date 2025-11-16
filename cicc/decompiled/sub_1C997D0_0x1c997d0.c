// Function: sub_1C997D0
// Address: 0x1c997d0
//
void __fastcall sub_1C997D0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // rbx
  int v7; // r9d
  _QWORD *v8; // r15
  unsigned __int8 v9; // al
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // r13
  unsigned __int64 v13; // r8
  _QWORD *v14; // rax
  int v15; // ecx
  _BYTE *v16; // rdx
  __int64 v17; // rsi
  unsigned __int64 v18; // r13
  unsigned __int64 *v19; // rax
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  _BOOL8 v23; // rdi
  unsigned __int64 *v25; // [rsp+18h] [rbp-98h]
  int v26; // [rsp+18h] [rbp-98h]
  _BYTE *v29; // [rsp+30h] [rbp-80h] BYREF
  __int64 v30; // [rsp+38h] [rbp-78h]
  _BYTE v31[112]; // [rsp+40h] [rbp-70h] BYREF

  for ( i = *(_QWORD *)(a3 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v8 = sub_1648700(i);
    v9 = *((_BYTE *)v8 + 16);
    if ( v9 <= 0x17u )
      BUG();
    if ( v9 == 56 )
    {
      v10 = *((_DWORD *)v8 + 5);
      v29 = v31;
      v30 = 0x800000000LL;
      v11 = 1LL - (v10 & 0xFFFFFFF);
      v12 = &v8[3 * v11];
      v13 = 0xAAAAAAAAAAAAAAABLL * ((-24 * v11) >> 3);
      if ( (unsigned __int64)(-24 * v11) > 0xC0 )
      {
        v26 = -1431655765 * ((-24 * v11) >> 3);
        sub_16CD150((__int64)&v29, v31, 0xAAAAAAAAAAAAAAABLL * ((-24 * v11) >> 3), 8, v13, v7);
        v16 = v29;
        v15 = v30;
        LODWORD(v13) = v26;
        v14 = &v29[8 * (unsigned int)v30];
      }
      else
      {
        v14 = v31;
        v15 = 0;
        v16 = v31;
      }
      if ( v12 != v8 )
      {
        do
        {
          if ( v14 )
            *v14 = *v12;
          v12 += 3;
          ++v14;
        }
        while ( v8 != v12 );
        v16 = v29;
        v15 = v30;
      }
      v17 = v8[7];
      LODWORD(v30) = v13 + v15;
      v18 = sub_15A9FF0(*(_QWORD *)(a1 + 352), v17, v16, (unsigned int)(v13 + v15)) + a4;
      v19 = (unsigned __int64 *)sub_22077B0(56);
      v19[4] = (unsigned __int64)v8;
      v19[6] = v18;
      v19[5] = a2;
      v25 = v19;
      v20 = sub_1C99730(a5, v19 + 4);
      if ( v21 )
      {
        v22 = a5 + 8;
        v23 = 1;
        if ( !v20 && v21 != v22 )
          v23 = (unsigned __int64)v8 < *(_QWORD *)(v21 + 32);
        sub_220F040(v23, v25, v21, v22);
        ++*(_QWORD *)(a5 + 40);
      }
      else
      {
        j_j___libc_free_0(v25, 56);
      }
      sub_1C997D0(a1, a2, v8, v18, a5);
      if ( v29 != v31 )
        _libc_free((unsigned __int64)v29);
    }
  }
}

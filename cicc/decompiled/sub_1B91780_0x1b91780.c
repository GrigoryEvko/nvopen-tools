// Function: sub_1B91780
// Address: 0x1b91780
//
void __fastcall sub_1B91780(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  int v6; // edx
  __int64 *v8; // rsi
  unsigned int v9; // ecx
  __int64 v10; // rax
  int v11; // edi
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r8
  __int64 v15; // rcx
  _BYTE *i; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-68h]
  _BYTE *v19; // [rsp+10h] [rbp-60h] BYREF
  __int64 v20; // [rsp+18h] [rbp-58h]
  _BYTE v21[80]; // [rsp+20h] [rbp-50h] BYREF

  v6 = 0;
  v8 = (__int64 *)v21;
  v20 = 0x400000000LL;
  v9 = *(_DWORD *)(a1 + 40);
  v10 = *(_QWORD *)(a1 + 24);
  v11 = *(_DWORD *)(a1 + 32);
  v19 = v21;
  if ( v11 )
  {
    v12 = v10 + 16LL * v9;
    if ( v10 != v12 )
    {
      while ( 1 )
      {
        v13 = v10;
        if ( (unsigned int)(*(_DWORD *)v10 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          break;
        v10 += 16;
        if ( v12 == v10 )
          goto LABEL_8;
      }
      if ( v12 == v10 )
      {
LABEL_8:
        v6 = 0;
        v8 = (__int64 *)v21;
        goto LABEL_2;
      }
      v14 = *(_QWORD *)(v10 + 8);
      v15 = 0;
      for ( i = v21; ; i = v19 )
      {
        *(_QWORD *)&i[8 * v15] = v14;
        v15 = (unsigned int)(v20 + 1);
        v17 = v13 + 16;
        LODWORD(v20) = v20 + 1;
        if ( v12 == v13 + 16 )
          break;
        while ( 1 )
        {
          v13 = v17;
          if ( (unsigned int)(*(_DWORD *)v17 + 0x7FFFFFFF) <= 0xFFFFFFFD )
            break;
          v17 += 16;
          if ( v12 == v17 )
            goto LABEL_14;
        }
        if ( v12 == v17 )
          break;
        v14 = *(_QWORD *)(v17 + 8);
        if ( HIDWORD(v20) <= (unsigned int)v15 )
        {
          v18 = *(_QWORD *)(v17 + 8);
          sub_16CD150((__int64)&v19, v21, 0, 8, v14, a6);
          v15 = (unsigned int)v20;
          v14 = v18;
        }
      }
LABEL_14:
      v8 = (__int64 *)v19;
      v6 = v15;
    }
  }
LABEL_2:
  sub_14C4AD0(a2, v8, v6);
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
}

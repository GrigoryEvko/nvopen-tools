// Function: sub_155FBC0
// Address: 0x155fbc0
//
__int64 __fastcall sub_155FBC0(__int64 *a1, int *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdx
  int *v5; // r15
  int *v6; // r13
  int v7; // r14d
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // r11
  int *v13; // rax
  int *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // [rsp-110h] [rbp-110h]
  __int64 v17; // [rsp-100h] [rbp-100h]
  _QWORD *v18; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v19; // [rsp-F0h] [rbp-F0h]
  _BYTE v20[32]; // [rsp-E8h] [rbp-E8h] BYREF
  int *v21; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v22; // [rsp-C0h] [rbp-C0h]
  _BYTE v23[184]; // [rsp-B8h] [rbp-B8h] BYREF

  if ( !a3 )
    return 0;
  v4 = 4 * a3;
  v5 = &a2[v4];
  v6 = a2;
  v21 = (int *)v23;
  v22 = 0x800000000LL;
  if ( a2 == &a2[v4] )
  {
    v14 = (int *)v23;
    v15 = 0;
  }
  else
  {
    do
    {
      v18 = v20;
      v7 = *v6;
      v8 = 0;
      v19 = 0x400000000LL;
      while ( 1 )
      {
        v9 = *((_QWORD *)v6 + 1);
        v6 += 4;
        v18[v8] = v9;
        v8 = (unsigned int)(v19 + 1);
        LODWORD(v19) = v19 + 1;
        if ( v6 == v5 || v7 != *v6 )
          break;
        if ( (unsigned int)v8 >= HIDWORD(v19) )
        {
          sub_16CD150(&v18, v20, 0, 8);
          v8 = (unsigned int)v19;
        }
      }
      v10 = sub_155F1F0(a1, v18, v8);
      v11 = v22;
      v12 = v10;
      if ( (unsigned int)v22 >= HIDWORD(v22) )
      {
        v16 = v10;
        sub_16CD150(&v21, v23, 0, 16);
        v11 = v22;
        v12 = v16;
      }
      v13 = &v21[4 * v11];
      if ( v13 )
      {
        *v13 = v7;
        *((_QWORD *)v13 + 1) = v12;
        v11 = v22;
      }
      LODWORD(v22) = v11 + 1;
      if ( v18 != (_QWORD *)v20 )
        _libc_free((unsigned __int64)v18);
    }
    while ( v5 != v6 );
    v14 = v21;
    v15 = (unsigned int)v22;
  }
  result = sub_155FA70(a1, v14, v15);
  if ( v21 != (int *)v23 )
  {
    v17 = result;
    _libc_free((unsigned __int64)v21);
    return v17;
  }
  return result;
}

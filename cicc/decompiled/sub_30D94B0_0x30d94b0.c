// Function: sub_30D94B0
// Address: 0x30d94b0
//
__int64 __fastcall sub_30D94B0(__int64 **a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r13d
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 **v10; // r10
  __int64 v11; // r14
  unsigned __int8 *v12; // r15
  unsigned __int8 *v13; // r14
  __int64 v14; // r14
  __int64 v15; // r8
  __int64 v16; // rbx
  unsigned __int8 **v17; // rcx
  int v18; // eax
  unsigned __int8 **v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // r12
  int v22; // edx
  int v23; // ebx
  __int64 v25; // rax
  int v26; // edx
  __int64 **v27; // [rsp+8h] [rbp-78h]
  unsigned __int8 **v28; // [rsp+20h] [rbp-60h] BYREF
  __int64 v29; // [rsp+28h] [rbp-58h]
  _BYTE v30[80]; // [rsp+30h] [rbp-50h] BYREF

  v7 = sub_30D92D0((__int64)a1, (__int64)a2, a3, a4, a5, a6);
  if ( !(_BYTE)v7 )
  {
    v9 = sub_30D1740((__int64)a1, *((_QWORD *)a2 - 4));
    if ( v9 )
      sub_30D1890((__int64)a1, v9);
    if ( (unsigned int)*a2 - 70 <= 5 )
    {
      v25 = sub_DFAF50(a1[1], *((_QWORD *)a2 + 1), v8);
      if ( !v26 && v25 == 4 )
        ((void (__fastcall *)(__int64 **))(*a1)[12])(a1);
    }
    v10 = (__int64 **)a1[1];
    v11 = 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
    if ( (a2[7] & 0x40) != 0 )
    {
      v12 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      v13 = &v12[v11];
    }
    else
    {
      v12 = &a2[-v11];
      v13 = a2;
    }
    v14 = v13 - v12;
    v28 = (unsigned __int8 **)v30;
    v15 = v14 >> 5;
    v29 = 0x400000000LL;
    v16 = v14 >> 5;
    if ( (unsigned __int64)v14 > 0x80 )
    {
      v27 = v10;
      sub_C8D5F0((__int64)&v28, v30, v14 >> 5, 8u, v15, (__int64)v30);
      v19 = v28;
      v18 = v29;
      v15 = v14 >> 5;
      v10 = v27;
      v17 = &v28[(unsigned int)v29];
    }
    else
    {
      v17 = (unsigned __int8 **)v30;
      v18 = 0;
      v19 = (unsigned __int8 **)v30;
    }
    if ( v14 > 0 )
    {
      v20 = 0;
      do
      {
        v17[v20 / 8] = *(unsigned __int8 **)&v12[4 * v20];
        v20 += 8LL;
        --v16;
      }
      while ( v16 );
      v19 = v28;
      v18 = v29;
    }
    LODWORD(v29) = v15 + v18;
    v21 = sub_DFCEF0(v10, a2, v19, (unsigned int)(v15 + v18), 3);
    v23 = v22;
    if ( v28 != (unsigned __int8 **)v30 )
      _libc_free((unsigned __int64)v28);
    if ( !v23 )
      LOBYTE(v7) = v21 == 0;
  }
  return v7;
}

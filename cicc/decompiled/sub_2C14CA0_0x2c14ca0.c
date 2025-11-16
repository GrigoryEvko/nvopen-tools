// Function: sub_2C14CA0
// Address: 0x2c14ca0
//
__int64 __fastcall sub_2C14CA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // ebx
  __int64 v8; // r8
  _DWORD *v9; // rdx
  _BYTE *v10; // rcx
  _DWORD *v11; // rax
  int v12; // ecx
  _QWORD **v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r13
  _BYTE *v19; // [rsp+10h] [rbp-70h] BYREF
  __int64 v20; // [rsp+18h] [rbp-68h]
  _BYTE v21[96]; // [rsp+20h] [rbp-60h] BYREF

  v7 = a2;
  if ( BYTE4(a2) )
  {
    if ( (_DWORD)a2 == 1 )
      return 0;
LABEL_3:
    v19 = v21;
    v20 = 0xC00000000LL;
    if ( !(_DWORD)a2 )
    {
LABEL_13:
      v13 = (_QWORD **)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
        v13 = (_QWORD **)**v13;
      v14 = sub_2BFD6A0(a3 + 16, (__int64)v13);
      v15 = sub_2AAEDF0(v14, a2);
      v16 = sub_DFBC30(
              *(__int64 **)a3,
              8,
              v15,
              (__int64)v19,
              (unsigned int)v20,
              *(unsigned int *)(a3 + 176),
              v7 - 1,
              0,
              0,
              0,
              0);
      if ( v19 != v21 )
        _libc_free((unsigned __int64)v19);
      return v16;
    }
    if ( (unsigned int)a2 > 0xCuLL )
    {
      sub_C8D5F0((__int64)&v19, v21, (unsigned int)a2, 4u, (unsigned int)a2, a6);
      v11 = v19;
      v8 = 4LL * (unsigned int)a2;
      v9 = &v19[4 * (unsigned int)v20];
      v10 = &v19[v8];
      if ( &v19[v8] == (_BYTE *)v9 )
      {
LABEL_10:
        LODWORD(v20) = a2;
        if ( v11 != v9 )
        {
          v12 = a2 - 1;
          do
            *v11++ = v12++;
          while ( v11 != v9 );
        }
        goto LABEL_13;
      }
    }
    else
    {
      v8 = 4LL * (unsigned int)a2;
      v9 = v21;
      v10 = &v21[v8];
    }
    do
    {
      if ( v9 )
        *v9 = 0;
      ++v9;
    }
    while ( v9 != (_DWORD *)v10 );
    v11 = v19;
    v9 = &v19[v8];
    goto LABEL_10;
  }
  if ( (_DWORD)a2 != 1 )
    goto LABEL_3;
  return sub_DFD270(*(_QWORD *)a3, 55, *(_DWORD *)(a3 + 176));
}

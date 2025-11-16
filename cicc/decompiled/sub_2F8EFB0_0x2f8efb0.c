// Function: sub_2F8EFB0
// Address: 0x2f8efb0
//
void __fastcall sub_2F8EFB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // r13
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  unsigned __int64 v13; // r12
  _QWORD *v14; // [rsp-88h] [rbp-88h] BYREF
  __int64 v15; // [rsp-80h] [rbp-80h]
  _QWORD v16[15]; // [rsp-78h] [rbp-78h] BYREF

  if ( (*(_BYTE *)(a1 + 254) & 1) != 0 )
  {
    v14 = v16;
    v16[0] = a1;
    v6 = v16;
    v15 = 0x800000001LL;
    LODWORD(v7) = 1;
    do
    {
      v8 = (unsigned int)v7;
      v7 = (unsigned int)(v7 - 1);
      v9 = v6[v8 - 1];
      LODWORD(v15) = v7;
      v10 = *(unsigned int *)(v9 + 128);
      v11 = *(_QWORD **)(v9 + 120);
      *(_BYTE *)(v9 + 254) &= ~1u;
      v12 = &v11[2 * v10];
      if ( v11 != v12 )
      {
        do
        {
          while ( 1 )
          {
            v13 = *v11 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v13 + 254) & 1) != 0 )
              break;
            v11 += 2;
            if ( v12 == v11 )
              goto LABEL_10;
          }
          if ( v7 + 1 > (unsigned __int64)HIDWORD(v15) )
          {
            sub_C8D5F0((__int64)&v14, v16, v7 + 1, 8u, a5, a6);
            v7 = (unsigned int)v15;
          }
          v11 += 2;
          v14[v7] = v13;
          v7 = (unsigned int)(v15 + 1);
          LODWORD(v15) = v15 + 1;
        }
        while ( v12 != v11 );
LABEL_10:
        v6 = v14;
      }
    }
    while ( (_DWORD)v7 );
    if ( v6 != v16 )
      _libc_free((unsigned __int64)v6);
  }
}

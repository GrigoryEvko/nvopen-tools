// Function: sub_2DE6A20
// Address: 0x2de6a20
//
void __fastcall sub_2DE6A20(char *dest, unsigned int a2)
{
  __int64 v3; // r8
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v6; // ebx
  unsigned int v7; // esi
  int v8; // ecx
  __int64 v9; // r12
  size_t v10; // rdx
  _BYTE *v11; // r12
  unsigned int v12; // [rsp-94h] [rbp-94h]
  _QWORD *v13; // [rsp-88h] [rbp-88h] BYREF
  __int64 v14; // [rsp-80h] [rbp-80h]
  _BYTE v15[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( a2 != 2 )
  {
    sub_2DE6A20(dest);
    sub_2DE6A20(&dest[8 * (a2 >> 1)]);
    v13 = v15;
    v14 = 0x800000000LL;
    if ( a2 )
    {
      v3 = a2 >> 1;
      v4 = 8;
      v5 = 0;
      v6 = 0;
      v7 = 0;
      v8 = 0;
      while ( 1 )
      {
        v9 = *(_QWORD *)&dest[8 * v7 + 8 * v8];
        if ( v5 + 1 > v4 )
        {
          v12 = v3;
          sub_C8D5F0((__int64)&v13, v15, v5 + 1, 8u, v3, v5 + 1);
          v5 = (unsigned int)v14;
          v3 = v12;
        }
        ++v6;
        v13[v5] = v9;
        v5 = (unsigned int)(v14 + 1);
        LODWORD(v14) = v14 + 1;
        if ( a2 == v6 )
          break;
        v4 = HIDWORD(v14);
        v7 = v6 >> 1;
        v8 = v6 & 1;
        if ( (v6 & 1) != 0 )
          v8 = v3;
      }
      v10 = 8 * v5;
      v11 = v13;
      if ( v10 )
        memmove(dest, v13, v10);
      if ( v11 != v15 )
        _libc_free((unsigned __int64)v11);
    }
  }
}

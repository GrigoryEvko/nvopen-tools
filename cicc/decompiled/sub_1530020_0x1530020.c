// Function: sub_1530020
// Address: 0x1530020
//
void __fastcall sub_1530020(__int64 ***a1)
{
  __int64 **v2; // rdi
  unsigned __int64 v3; // rdi
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 *v8; // rax
  __int64 v9; // r10
  signed __int64 v10; // r8
  _BYTE *v11; // rcx
  __int64 i; // rax
  __int64 **v13; // rdi
  __int64 v14; // [rsp+10h] [rbp-2E0h]
  signed __int64 v15; // [rsp+18h] [rbp-2D8h]
  _BYTE *v16; // [rsp+20h] [rbp-2D0h] BYREF
  __int64 v17; // [rsp+28h] [rbp-2C8h]
  _BYTE v18[128]; // [rsp+30h] [rbp-2C0h] BYREF
  _BYTE *v19; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v20; // [rsp+B8h] [rbp-238h]
  _BYTE v21[560]; // [rsp+C0h] [rbp-230h] BYREF

  v2 = a1[2];
  v20 = 0x4000000000LL;
  v17 = 0x800000000LL;
  v19 = v21;
  v16 = v18;
  sub_1632060(v2, &v16);
  if ( !(_DWORD)v17 )
  {
    v3 = (unsigned __int64)v16;
    if ( v16 == v18 )
      goto LABEL_4;
    goto LABEL_3;
  }
  sub_1526BE0(*a1, 0x16u, 3u);
  v4 = (unsigned int)v17;
  if ( (_DWORD)v17 )
  {
    v5 = (unsigned int)v20;
    v6 = 0;
    do
    {
      if ( HIDWORD(v20) <= (unsigned int)v5 )
      {
        sub_16CD150(&v19, v21, 0, 8);
        v5 = (unsigned int)v20;
      }
      *(_QWORD *)&v19[8 * v5] = v6;
      v8 = (__int64 *)&v16[16 * v6];
      LODWORD(v20) = v20 + 1;
      v7 = (unsigned int)v20;
      v9 = *v8;
      v10 = v8[1];
      if ( v10 > HIDWORD(v20) - (unsigned __int64)(unsigned int)v20 )
      {
        v14 = *v8;
        v15 = v8[1];
        sub_16CD150(&v19, v21, v10 + (unsigned int)v20, 8);
        v7 = (unsigned int)v20;
        v9 = v14;
        v10 = v15;
      }
      v11 = &v19[8 * v7];
      if ( v10 > 0 )
      {
        for ( i = 0; i != v10; ++i )
          *(_QWORD *)&v11[8 * i] = *(char *)(v9 + i);
        LODWORD(v7) = v20;
      }
      v13 = *a1;
      ++v6;
      LODWORD(v20) = v7 + v10;
      sub_152F3D0(v13, 6u, (__int64)&v19, 0);
      v5 = 0;
      LODWORD(v20) = 0;
    }
    while ( v4 != v6 );
  }
  sub_15263C0(*a1);
  v3 = (unsigned __int64)v16;
  if ( v16 != v18 )
LABEL_3:
    _libc_free(v3);
LABEL_4:
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
}

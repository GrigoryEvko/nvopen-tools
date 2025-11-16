// Function: sub_1AA7010
// Address: 0x1aa7010
//
__int64 __fastcall sub_1AA7010(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rbx
  int v6; // eax
  unsigned __int64 *v7; // rdi
  unsigned __int64 v8; // rax
  bool v9; // zf
  __int64 v10; // rax
  _BYTE *v11; // r13
  __int64 v12; // rbx
  unsigned int v13; // r12d
  __int64 v14; // rdx
  __int64 v15; // rdi
  int v16; // eax
  _QWORD *v17; // rbx
  __int64 v18; // rax
  __int64 v20; // [rsp+8h] [rbp-128h]
  _QWORD v21[2]; // [rsp+10h] [rbp-120h] BYREF
  unsigned __int64 v22; // [rsp+20h] [rbp-110h]
  _BYTE *v23; // [rsp+30h] [rbp-100h] BYREF
  __int64 v24; // [rsp+38h] [rbp-F8h]
  _BYTE v25[240]; // [rsp+40h] [rbp-F0h] BYREF

  v23 = v25;
  v24 = 0x800000000LL;
  v2 = sub_157F280(a1);
  v4 = v3;
  v5 = v2;
  while ( v4 != v5 )
  {
    v22 = v5;
    v21[0] = 6;
    v21[1] = 0;
    if ( v5 != -8 && v5 != 0 && v5 != -16 )
      sub_164C220((__int64)v21);
    v6 = v24;
    if ( (unsigned int)v24 >= HIDWORD(v24) )
    {
      sub_170B450((__int64)&v23, 0);
      v6 = v24;
    }
    v7 = (unsigned __int64 *)&v23[24 * v6];
    if ( v7 )
    {
      *v7 = 6;
      v7[1] = 0;
      v8 = v22;
      v9 = v22 == -8;
      v7[2] = v22;
      if ( v8 != 0 && !v9 && v8 != -16 )
        sub_1649AC0(v7, v21[0] & 0xFFFFFFFFFFFFFFF8LL);
      v6 = v24;
    }
    LODWORD(v24) = v6 + 1;
    if ( v22 != -8 && v22 != 0 && v22 != -16 )
      sub_1649B30(v21);
    if ( !v5 )
      BUG();
    v10 = *(_QWORD *)(v5 + 32);
    if ( !v10 )
      BUG();
    v5 = 0;
    if ( *(_BYTE *)(v10 - 8) == 77 )
      v5 = v10 - 24;
  }
  v11 = v23;
  if ( (_DWORD)v24 )
  {
    v12 = 0;
    v13 = 0;
    v14 = 24LL * (unsigned int)v24;
    do
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)&v11[v12 + 16];
        if ( v15 )
        {
          if ( *(_BYTE *)(v15 + 16) == 77 )
            break;
        }
        v12 += 24;
        if ( v14 == v12 )
          goto LABEL_26;
      }
      v20 = v14;
      v12 += 24;
      v16 = sub_1AEB420(v15, a2);
      v14 = v20;
      v11 = v23;
      v13 |= v16;
    }
    while ( v20 != v12 );
LABEL_26:
    v17 = &v11[24 * (unsigned int)v24];
    if ( v17 != (_QWORD *)v11 )
    {
      do
      {
        v18 = *(v17 - 1);
        v17 -= 3;
        if ( v18 != -8 && v18 != 0 && v18 != -16 )
          sub_1649B30(v17);
      }
      while ( v17 != (_QWORD *)v11 );
      v11 = v23;
    }
  }
  else
  {
    v13 = 0;
  }
  if ( v11 != v25 )
    _libc_free((unsigned __int64)v11);
  return v13;
}

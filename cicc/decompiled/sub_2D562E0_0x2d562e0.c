// Function: sub_2D562E0
// Address: 0x2d562e0
//
__int64 __fastcall sub_2D562E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 *v10; // r12
  unsigned int v11; // eax
  unsigned __int64 v12; // rbx
  unsigned __int64 *v13; // r12
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rdi
  int v16; // eax
  unsigned __int64 v17; // rbx
  unsigned __int64 *v18; // r13
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // rdi
  __int64 *v21; // [rsp+0h] [rbp-E0h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-D8h]
  char v23; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int64 v24[24]; // [rsp+20h] [rbp-C0h] BYREF

  v6 = 0;
  sub_2D55110((__int64)&v21, a2, a3, a4, a5, a6);
  if ( v22 )
  {
    memset(v24, 0, 0x88u);
    v8 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
    if ( v8 && (v9 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v8 + 104LL))(v8, &unk_4F8144C)) != 0 )
    {
      v10 = (unsigned __int64 *)(v9 + 176);
    }
    else
    {
      if ( LOBYTE(v24[16]) )
      {
        v17 = v24[3];
        LOBYTE(v24[16]) = 0;
        v18 = (unsigned __int64 *)(v24[3] + 8LL * LODWORD(v24[4]));
        if ( (unsigned __int64 *)v24[3] != v18 )
        {
          do
          {
            v19 = *--v18;
            if ( v19 )
            {
              v20 = *(_QWORD *)(v19 + 24);
              if ( v20 != v19 + 40 )
                _libc_free(v20);
              j_j___libc_free_0(v19);
            }
          }
          while ( (unsigned __int64 *)v17 != v18 );
          v18 = (unsigned __int64 *)v24[3];
        }
        if ( v18 != &v24[5] )
          _libc_free((unsigned __int64)v18);
        if ( (unsigned __int64 *)v24[0] != &v24[2] )
          _libc_free(v24[0]);
      }
      v24[13] = a2;
      v24[1] = 0x100000000LL;
      v24[4] = 0x600000000LL;
      v16 = *(_DWORD *)(a2 + 92);
      v10 = v24;
      v24[0] = (unsigned __int64)&v24[2];
      v24[3] = (unsigned __int64)&v24[5];
      v24[12] = 0;
      LOBYTE(v24[14]) = 0;
      HIDWORD(v24[14]) = 0;
      LODWORD(v24[15]) = v16;
      sub_B1F440((__int64)v24);
      LOBYTE(v24[16]) = 1;
    }
    v6 = sub_2D552F0(v21, v22, (__int64)v10);
    v11 = sub_2D557B0(v21, v22, (__int64)v10);
    if ( (_BYTE)v11 )
      v6 = v11;
    if ( LOBYTE(v24[16]) )
    {
      v12 = v24[3];
      LOBYTE(v24[16]) = 0;
      v13 = (unsigned __int64 *)(v24[3] + 8LL * LODWORD(v24[4]));
      if ( (unsigned __int64 *)v24[3] != v13 )
      {
        do
        {
          v14 = *--v13;
          if ( v14 )
          {
            v15 = *(_QWORD *)(v14 + 24);
            if ( v15 != v14 + 40 )
              _libc_free(v15);
            j_j___libc_free_0(v14);
          }
        }
        while ( (unsigned __int64 *)v12 != v13 );
        v13 = (unsigned __int64 *)v24[3];
      }
      if ( v13 != &v24[5] )
        _libc_free((unsigned __int64)v13);
      if ( (unsigned __int64 *)v24[0] != &v24[2] )
        _libc_free(v24[0]);
    }
  }
  if ( v21 != (__int64 *)&v23 )
    _libc_free((unsigned __int64)v21);
  return v6;
}

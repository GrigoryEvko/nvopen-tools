// Function: sub_866010
// Address: 0x866010
//
_DWORD *sub_866010()
{
  __int64 v0; // rdi
  __int64 v1; // rax
  int v2; // r12d
  int v3; // r15d
  char v4; // r14
  bool v5; // r13
  bool v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 *v10; // r9
  unsigned int v11; // edx
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 *v16; // r9
  int v17; // edi
  __int64 v18; // rdx
  __int64 v19; // rcx
  bool v21; // zf
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rcx
  char v25; // [rsp+Fh] [rbp-31h]

  v0 = dword_4F04C64;
  v1 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v2 = *(_DWORD *)(v1 + 572);
  v3 = *(_DWORD *)(v1 + 576);
  v4 = *(_BYTE *)(v1 + 8);
  v5 = (v4 & 0x40) != 0;
  v6 = (v4 & 0x10) != 0;
  v25 = *(_BYTE *)(v1 + 8) >> 7;
  sub_85FE80(dword_4F04C64, 0, 0);
  while ( dword_4F04C64 > v2 )
    sub_863FC0(v0, 0, v7, v8, v9, v10);
  v11 = 0;
  if ( unk_4F04C48 != -1 )
  {
    v11 = 0;
    if ( (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1) != 0 )
    {
      if ( dword_4D047C8 )
        v11 = sub_7D3BE0(v0, 0, 0, qword_4F04C68, v9);
    }
  }
  v12 = (unsigned int)dword_4F04C64;
  sub_85FE80(dword_4F04C64, 1, v11);
  if ( v5 )
  {
    if ( v25 )
    {
      v17 = dword_4F04C64;
      v18 = 776LL * dword_4F04C64;
      v19 = qword_4F04C68[0] + v18;
      if ( *(_BYTE *)(qword_4F04C68[0] + v18 + 4) == 8 )
      {
        v21 = *(_BYTE *)(qword_4F04C68[0] + v18 - 772) == 8;
        v22 = v18 - 776;
        v23 = dword_4F04C64;
        if ( v21 )
        {
          do
          {
            v24 = v22;
            v22 -= 776;
            --v23;
          }
          while ( *(_BYTE *)(qword_4F04C68[0] + v22 + 4) == 8 );
          v19 = qword_4F04C68[0] + v24;
        }
        *(_DWORD *)(v19 + 552) = v23 - 1;
        dword_4F04C60 = v17;
      }
    }
  }
  else if ( v6 )
  {
    if ( *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 4) == 4 )
      sub_8642D0(v12, (__int64)qword_4F04C68, v13, v14, v15, v16);
    else
      sub_8645D0(v12, (__int64)qword_4F04C68, v13, v14, v15, v16);
  }
  dword_4F04C2C = v3;
  return &dword_4F04C2C;
}

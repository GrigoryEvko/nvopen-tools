// Function: sub_301F0C0
// Address: 0x301f0c0
//
_QWORD *sub_301F0C0()
{
  _QWORD *v0; // r12
  __int64 v1; // r13
  unsigned __int64 v2; // rcx
  unsigned __int64 *v3; // r14
  unsigned __int64 v4; // rax
  unsigned __int64 *v6; // r13
  unsigned __int64 *v7; // rbx

  v0 = (_QWORD *)sub_22077B0(0xF8u);
  if ( v0 )
  {
    v1 = 0;
    memset(v0, 0, 0xF8u);
    v2 = 0;
    v3 = 0;
    *v0 = &unk_49E3560;
    v4 = 0;
  }
  else
  {
    v3 = (unsigned __int64 *)MEMORY[0xE8];
    v1 = MEMORY[0xE0];
    v4 = MEMORY[0xE8] - MEMORY[0xE0];
    v2 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(MEMORY[0xE8] - MEMORY[0xE0]) >> 3);
  }
  v0[2] = 103;
  *((_DWORD *)v0 + 6) = 0;
  v0[1] = &unk_4458100;
  v0[5] = 0x660000000DLL;
  v0[4] = &off_49D44C0;
  *((_DWORD *)v0 + 24) = 1;
  v0[7] = &unk_4458D10;
  v0[14] = 0;
  v0[8] = &unk_4458D00;
  v0[15] = 0;
  v0[9] = "ENVREG10";
  v0[16] = 0;
  v0[10] = "Int1Regs";
  v0[17] = 0;
  v0[6] = &unk_4457F60;
  v0[18] = 0;
  v0[11] = &unk_4458CF2;
  v0[19] = 0;
  v0[13] = &unk_4457C40;
  if ( v4 <= 0x990 )
  {
    sub_301EE80((__int64)(v0 + 28), 103 - v2);
    return v0;
  }
  else
  {
    if ( v4 <= 0x9A8 )
      return v0;
    v6 = (unsigned __int64 *)(v1 + 2472);
    if ( v6 == v3 )
    {
      return v0;
    }
    else
    {
      v7 = v6;
      do
      {
        if ( *v7 )
          j_j___libc_free_0(*v7);
        v7 += 3;
      }
      while ( v3 != v7 );
      v0[29] = v6;
      return v0;
    }
  }
}

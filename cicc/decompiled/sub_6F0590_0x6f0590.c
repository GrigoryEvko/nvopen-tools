// Function: sub_6F0590
// Address: 0x6f0590
//
_QWORD *sub_6F0590()
{
  __int64 *v0; // rbx
  _DWORD *v1; // rax
  _DWORD *v2; // rcx
  _DWORD *v3; // rdx
  _QWORD *v4; // rbx
  _QWORD *v5; // rax
  _QWORD *v6; // rcx
  _QWORD *v7; // rdx
  _QWORD *v8; // rbx
  _QWORD *result; // rax
  _QWORD *v10; // rcx
  _QWORD *v11; // rdx

  qword_4D03A68 = 0;
  qword_4D03A60 = 0;
  qword_4D03C50 = 0;
  qword_4D03A58 = 0;
  unk_4D03C48 = 0;
  unk_4D03C40 = 0;
  dword_4D03C08[0] = 0;
  unk_4D03C78 = 0;
  unk_4D03C70 = 0;
  unk_4D03C20 = 0;
  v0 = (__int64 *)sub_823970(16);
  qword_4D03C00 = v0;
  if ( v0 )
  {
    v1 = (_DWORD *)sub_823970(24576);
    v2 = v1;
    v3 = v1 + 6144;
    do
    {
      if ( v1 )
        *v1 = 0;
      v1 += 6;
    }
    while ( v3 != v1 );
    *v0 = (__int64)v2;
    v0[1] = 1023;
  }
  v4 = (_QWORD *)sub_823970(16);
  unk_4D03BF8 = v4;
  if ( v4 )
  {
    v5 = (_QWORD *)sub_823970(0x2000);
    v6 = v5;
    v7 = v5 + 1024;
    do
    {
      if ( v5 )
        *v5 = 0;
      v5 += 4;
    }
    while ( v7 != v5 );
    *v4 = v6;
    v4[1] = 255;
  }
  v8 = (_QWORD *)sub_823970(16);
  result = &qword_4D03BF0;
  qword_4D03BF0 = v8;
  if ( v8 )
  {
    result = (_QWORD *)sub_823970(512);
    v10 = result;
    v11 = result + 64;
    do
    {
      if ( result )
        *result = 0;
      result += 2;
    }
    while ( v11 != result );
    *v8 = v10;
    v8[1] = 31;
  }
  return result;
}

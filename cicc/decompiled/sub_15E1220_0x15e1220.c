// Function: sub_15E1220
// Address: 0x15e1220
//
void __fastcall sub_15E1220(int a1, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // rax
  _BYTE *v4; // rdx
  char v5; // r14
  _BYTE *v6; // r15
  __int64 v7; // rbx
  __int64 i; // rax
  int v9; // [rsp+Ch] [rbp-54h] BYREF
  _BYTE *v10; // [rsp+10h] [rbp-50h] BYREF
  __int64 v11; // [rsp+18h] [rbp-48h]
  _BYTE v12[64]; // [rsp+20h] [rbp-40h] BYREF

  v2 = dword_42A3340[a1 - 1];
  v10 = v12;
  v11 = 0x800000000LL;
  if ( v2 < 0 )
  {
    v6 = &unk_42A1140;
    v9 = v2 & 0x7FFFFFFF;
    v7 = 8682;
  }
  else
  {
    v9 = 0;
    v3 = 0;
    v4 = v12;
    v5 = v2 & 0xF;
    while ( 1 )
    {
      v4[v3] = v5;
      v3 = (unsigned int)(v11 + 1);
      v2 = (unsigned int)v2 >> 4;
      LODWORD(v11) = v11 + 1;
      if ( !v2 )
        break;
      v5 = v2 & 0xF;
      if ( HIDWORD(v11) <= (unsigned int)v3 )
      {
        sub_16CD150(&v10, v12, 0, 1);
        v3 = (unsigned int)v11;
      }
      v4 = v10;
    }
    v9 = 0;
    v6 = v10;
    v7 = (unsigned int)v3;
  }
  sub_15DEB50((unsigned int *)&v9, (__int64)v6, v7, a2);
  for ( i = (unsigned int)v9; v9 != v7; i = (unsigned int)v9 )
  {
    if ( !v6[i] )
      break;
    sub_15DEB50((unsigned int *)&v9, (__int64)v6, v7, a2);
  }
  if ( v10 != v12 )
    _libc_free((unsigned __int64)v10);
}

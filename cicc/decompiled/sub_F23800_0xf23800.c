// Function: sub_F23800
// Address: 0xf23800
//
__int64 __fastcall sub_F23800(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  unsigned __int8 *v3; // rax
  __int64 v4; // r12
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r13
  __int64 v11; // rbx

  v2 = 0;
  while ( 1 )
  {
    v3 = (unsigned __int8 *)sub_B46BC0(a2, 0);
    v4 = (__int64)v3;
    if ( !v3 )
      break;
    v5 = (unsigned int)*v3 - 39;
    if ( (unsigned int)v5 <= 0x38 )
    {
      v6 = 0x100060000000001LL;
      if ( _bittest64(&v6, v5) )
        break;
    }
    v7 = sub_98CD80((char *)v4);
    if ( !(_BYTE)v7 )
      break;
    v8 = sub_ACADE0(*(__int64 ***)(v4 + 8));
    v9 = *(_QWORD *)(v4 + 16);
    v10 = v8;
    if ( v9 )
    {
      v11 = *(_QWORD *)(a1 + 40);
      do
      {
        sub_F15FC0(v11, *(_QWORD *)(v9 + 24));
        v9 = *(_QWORD *)(v9 + 8);
      }
      while ( v9 );
      if ( v10 == v4 )
        v10 = sub_ACADE0(*(__int64 ***)(v10 + 8));
      if ( !*(_QWORD *)(v10 + 16)
        && *(_BYTE *)v10 > 0x1Cu
        && (*(_BYTE *)(v10 + 7) & 0x10) == 0
        && (*(_BYTE *)(v4 + 7) & 0x10) != 0 )
      {
        sub_BD6B90((unsigned __int8 *)v10, (unsigned __int8 *)v4);
      }
      sub_BD84D0(v4, v10);
    }
    v2 = v7;
    sub_F207A0(a1, (__int64 *)v4);
  }
  return v2;
}

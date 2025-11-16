// Function: sub_2667650
// Address: 0x2667650
//
void __fastcall sub_2667650(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned __int8 *v8; // rdi
  int v9; // eax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax

  v3 = 0x8000000000041LL;
  v6 = *(_QWORD *)(a2 + 16);
  while ( v6 )
  {
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 8);
    v8 = *(unsigned __int8 **)(v7 + 24);
    v9 = *v8;
    if ( (unsigned __int8)v9 > 0x1Cu )
    {
      v10 = (unsigned int)(v9 - 34);
      if ( (unsigned __int8)v10 <= 0x33u && _bittest64(&v3, v10) && (unsigned __int8 *)v7 == v8 - 32 )
      {
        v11 = sub_B43CB0((__int64)v8);
        sub_26673B0(a1, v11);
        if ( *(_QWORD *)v7 )
        {
          v12 = *(_QWORD *)(v7 + 8);
          **(_QWORD **)(v7 + 16) = v12;
          if ( v12 )
            *(_QWORD *)(v12 + 16) = *(_QWORD *)(v7 + 16);
        }
        *(_QWORD *)v7 = a3;
        if ( a3 )
        {
          v13 = *(_QWORD *)(a3 + 16);
          *(_QWORD *)(v7 + 8) = v13;
          if ( v13 )
            *(_QWORD *)(v13 + 16) = v7 + 8;
          *(_QWORD *)(v7 + 16) = a3 + 16;
          *(_QWORD *)(a3 + 16) = v7;
        }
      }
    }
  }
}

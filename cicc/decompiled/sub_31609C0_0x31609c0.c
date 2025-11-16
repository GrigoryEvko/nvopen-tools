// Function: sub_31609C0
// Address: 0x31609c0
//
void __fastcall sub_31609C0(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  _BYTE *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx

  v3 = *(_QWORD *)(a1 + 16);
  while ( v3 )
  {
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 8);
    v5 = *(_BYTE **)(v4 + 24);
    if ( (unsigned __int8)(*v5 - 61) > 1u )
    {
      v6 = sub_31604B0((__int64)v5, a1, a2);
      if ( *(_QWORD *)v4 )
      {
        v7 = *(_QWORD *)(v4 + 8);
        **(_QWORD **)(v4 + 16) = v7;
        if ( v7 )
          *(_QWORD *)(v7 + 16) = *(_QWORD *)(v4 + 16);
      }
      *(_QWORD *)v4 = v6;
      if ( v6 )
      {
        v8 = *(_QWORD *)(v6 + 16);
        *(_QWORD *)(v4 + 8) = v8;
        if ( v8 )
          *(_QWORD *)(v8 + 16) = v4 + 8;
        *(_QWORD *)(v4 + 16) = v6 + 16;
        *(_QWORD *)(v6 + 16) = v4;
      }
    }
  }
}

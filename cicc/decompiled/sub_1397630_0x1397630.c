// Function: sub_1397630
// Address: 0x1397630
//
void __fastcall sub_1397630(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // r14
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax

  v1 = *(_QWORD *)(a1 + 64);
  if ( v1 )
  {
    *(_DWORD *)(v1 + 32) = 0;
    v2 = *(_QWORD **)(a1 + 64);
    if ( v2 )
    {
      v3 = v2[2];
      v4 = v2[1];
      if ( v3 != v4 )
      {
        do
        {
          v5 = *(_QWORD *)(v4 + 16);
          if ( v5 != 0 && v5 != -8 && v5 != -16 )
            sub_1649B30(v4);
          v4 += 32;
        }
        while ( v3 != v4 );
        v4 = v2[1];
      }
      if ( v4 )
        j_j___libc_free_0(v4, v2[3] - v4);
      j_j___libc_free_0(v2, 40);
    }
  }
  sub_1396A40(*(_QWORD **)(a1 + 24));
}

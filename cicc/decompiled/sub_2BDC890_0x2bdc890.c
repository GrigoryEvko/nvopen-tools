// Function: sub_2BDC890
// Address: 0x2bdc890
//
void __fastcall sub_2BDC890(__int64 a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  void (__fastcall *v4)(unsigned __int64, unsigned __int64, __int64); // rax
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rdi

  sub_2209150((volatile signed __int32 **)(a1 + 96));
  v2 = *(_QWORD *)(a1 + 80);
  v3 = *(_QWORD *)(a1 + 72);
  if ( v2 != v3 )
  {
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v3 == 11 )
        {
          v4 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v3 + 32);
          if ( v4 )
            break;
        }
        v3 += 48LL;
        if ( v2 == v3 )
          goto LABEL_7;
      }
      v5 = v3 + 16;
      v3 += 48LL;
      v4(v5, v5, 3);
    }
    while ( v2 != v3 );
LABEL_7:
    v3 = *(_QWORD *)(a1 + 72);
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  v6 = *(_QWORD *)(a1 + 16);
  if ( v6 )
    j_j___libc_free_0(v6);
}

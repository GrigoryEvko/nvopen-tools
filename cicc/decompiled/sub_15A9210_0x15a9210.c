// Function: sub_15A9210
// Address: 0x15a9210
//
void __fastcall sub_15A9210(__int64 a1)
{
  __int64 v2; // r14
  _QWORD *v3; // rdi
  _QWORD *v4; // r12
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  unsigned __int64 v7; // rdi

  v2 = *(_QWORD *)(a1 + 400);
  *(_DWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  if ( v2 )
  {
    v3 = *(_QWORD **)(v2 + 8);
    if ( *(_DWORD *)(v2 + 16) )
    {
      v4 = &v3[2 * *(unsigned int *)(v2 + 24)];
      if ( v4 != v3 )
      {
        v5 = *(_QWORD **)(v2 + 8);
        while ( 1 )
        {
          v6 = v5;
          if ( *v5 != -8 && *v5 != -16 )
            break;
          v5 += 2;
          if ( v4 == v5 )
            goto LABEL_3;
        }
        if ( v4 != v5 )
        {
          do
          {
            v7 = v6[1];
            v6 += 2;
            _libc_free(v7);
            if ( v6 == v4 )
              break;
            while ( *v6 == -16 || *v6 == -8 )
            {
              v6 += 2;
              if ( v4 == v6 )
                goto LABEL_16;
            }
          }
          while ( v4 != v6 );
LABEL_16:
          v3 = *(_QWORD **)(v2 + 8);
        }
      }
    }
LABEL_3:
    j___libc_free_0(v3);
    j_j___libc_free_0(v2, 32);
  }
  *(_QWORD *)(a1 + 400) = 0;
}

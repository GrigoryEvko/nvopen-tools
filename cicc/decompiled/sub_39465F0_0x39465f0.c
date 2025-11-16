// Function: sub_39465F0
// Address: 0x39465f0
//
void __fastcall sub_39465F0(__int64 a1)
{
  unsigned __int64 v1; // r8
  __int64 v3; // rdx
  _QWORD *v4; // r12
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 v7; // rdi

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 16) )
  {
    v3 = *(unsigned int *)(a1 + 24);
    v4 = (_QWORD *)(v1 + 16 * v3);
    if ( v4 != (_QWORD *)v1 )
    {
      v5 = *(_QWORD **)(a1 + 8);
      while ( 1 )
      {
        v6 = v5;
        if ( *v5 != -8 && *v5 != -4 )
          break;
        v5 += 2;
        if ( v4 == v5 )
          goto LABEL_8;
      }
      if ( v4 != v5 )
      {
        do
        {
          v7 = v6[1];
          if ( v7 )
          {
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 32LL))(v7);
            v1 = *(_QWORD *)(a1 + 8);
            v3 = *(unsigned int *)(a1 + 24);
          }
          for ( v6 += 2; v4 != v6; v6 += 2 )
          {
            if ( *v6 != -4 && *v6 != -8 )
              break;
          }
        }
        while ( v6 != (_QWORD *)(v1 + 16 * v3) );
      }
    }
LABEL_8:
    j___libc_free_0(v1);
  }
  else
  {
    j___libc_free_0(*(_QWORD *)(a1 + 8));
  }
}

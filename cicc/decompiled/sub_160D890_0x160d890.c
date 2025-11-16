// Function: sub_160D890
// Address: 0x160d890
//
void __fastcall sub_160D890(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // r12
  _QWORD *v3; // rbx
  __int64 v4; // r14

  if ( a1 )
  {
    if ( *(_DWORD *)(a1 + 16) )
    {
      v1 = *(_QWORD **)(a1 + 8);
      v2 = &v1[2 * *(unsigned int *)(a1 + 24)];
      if ( v1 != v2 )
      {
        while ( 1 )
        {
          v3 = v1;
          if ( *v1 != -4 && *v1 != -8 )
            break;
          v1 += 2;
          if ( v2 == v1 )
            goto LABEL_3;
        }
        while ( v2 != v3 )
        {
          v4 = v3[1];
          if ( v4 )
          {
            sub_16D93B0(v3[1]);
            j_j___libc_free_0(v4, 160);
          }
          v3 += 2;
          if ( v3 == v2 )
            break;
          while ( *v3 == -8 || *v3 == -4 )
          {
            v3 += 2;
            if ( v2 == v3 )
              goto LABEL_3;
          }
        }
      }
    }
LABEL_3:
    sub_16D9420(a1 + 32);
    j___libc_free_0(*(_QWORD *)(a1 + 8));
    j_j___libc_free_0(a1, 144);
  }
}

// Function: sub_1688820
// Address: 0x1688820
//
__int64 __fastcall sub_1688820(__int64 a1, char *a2, size_t a3)
{
  char *v4; // r12
  __int64 result; // rax
  unsigned __int8 *v6; // r14
  int v7; // edi
  FILE *v8; // rcx

  v4 = a2;
  if ( a1 )
  {
    switch ( *(_DWORD *)a1 )
    {
      case 0:
        result = (*(__int64 (__fastcall **)(_QWORD, char *))(a1 + 8))(*(_QWORD *)(a1 + 32), a2);
        break;
      case 1:
        goto LABEL_6;
      case 2:
        sub_1688330(*(__int64 **)(a1 + 32), a2, a3);
        result = a3;
        break;
      case 3:
        v8 = *(FILE **)(a1 + 32);
        if ( v8 )
          return fwrite(a2, 1u, a3, v8);
        result = 0;
        if ( a3 )
        {
          v6 = (unsigned __int8 *)&a2[a3];
          do
          {
            v7 = (unsigned __int8)*v4++;
            _IO_putc(v7, stdout);
          }
          while ( v4 != (char *)v6 );
LABEL_6:
          result = a3;
        }
        break;
      case 4:
        memcpy(*(void **)(a1 + 32), a2, a3);
        *(_QWORD *)(a1 + 32) += a3;
        result = a3;
        break;
      default:
        result = -1;
        break;
    }
  }
  else
  {
    v8 = stdout;
    return fwrite(a2, 1u, a3, v8);
  }
  return result;
}

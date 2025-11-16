// Function: sub_D38990
// Address: 0xd38990
//
__int64 __fastcall sub_D38990(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rdi

  result = *((unsigned int *)a1 + 2);
  v7 = *a1;
  v8 = *a1 + 48 * result;
  if ( *a1 != v8 )
  {
    v9 = a2;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v7;
        v10 = *(_QWORD *)(v7 + 8);
        *(_DWORD *)(v9 + 24) = 0;
        *(_QWORD *)(v9 + 8) = v10;
        *(_QWORD *)(v9 + 16) = v9 + 32;
        *(_DWORD *)(v9 + 28) = 2;
        if ( *(_DWORD *)(v7 + 24) )
        {
          a2 = v7 + 16;
          sub_D323F0(v9 + 16, (char **)(v7 + 16), a3, a4, a5, a6);
        }
        *(_DWORD *)(v9 + 40) = *(_DWORD *)(v7 + 40);
        *(_BYTE *)(v9 + 44) = *(_BYTE *)(v7 + 44);
      }
      v7 += 48;
      v9 += 48;
    }
    while ( v8 != v7 );
    result = *((unsigned int *)a1 + 2);
    v11 = *a1;
    v12 = *a1 + 48 * result;
    if ( *a1 != v12 )
    {
      do
      {
        v12 -= 48;
        v13 = *(_QWORD *)(v12 + 16);
        result = v12 + 32;
        if ( v13 != v12 + 32 )
          result = _libc_free(v13, a2);
      }
      while ( v12 != v11 );
    }
  }
  return result;
}

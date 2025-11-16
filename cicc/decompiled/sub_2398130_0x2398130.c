// Function: sub_2398130
// Address: 0x2398130
//
void __fastcall sub_2398130(__int64 a1)
{
  __int64 v1; // r13
  _QWORD *v2; // rbx
  __int64 v3; // r13
  _QWORD *v4; // r13
  __int64 v5; // rax
  unsigned __int64 *v6; // rax
  unsigned __int64 v7; // r12

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v2 = (_QWORD *)(a1 + 16);
    v3 = 4;
  }
  else
  {
    v1 = *(unsigned int *)(a1 + 24);
    if ( !(_DWORD)v1 )
      return;
    v2 = *(_QWORD **)(a1 + 16);
    v3 = 2 * v1;
  }
  v4 = &v2[v3];
  do
  {
    if ( *v2 != -8192 && *v2 != -4096 )
    {
      v5 = v2[1];
      if ( v5 )
      {
        if ( (v5 & 4) != 0 )
        {
          v6 = (unsigned __int64 *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
          v7 = (unsigned __int64)v6;
          if ( v6 )
          {
            if ( (unsigned __int64 *)*v6 != v6 + 2 )
              _libc_free(*v6);
            j_j___libc_free_0(v7);
          }
        }
      }
    }
    v2 += 2;
  }
  while ( v2 != v4 );
}

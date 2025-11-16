// Function: sub_1B05570
// Address: 0x1b05570
//
__int64 __fastcall sub_1B05570(__int64 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rax
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // r13

  v1 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v1 )
  {
    v2 = *(_QWORD **)(a1 + 8);
    v3 = &v2[2 * v1];
    do
    {
      if ( *v2 != -16 && *v2 != -8 )
      {
        v4 = v2[1];
        if ( (v4 & 4) != 0 )
        {
          v5 = (unsigned __int64 *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
          v6 = v5;
          if ( v5 )
          {
            if ( (unsigned __int64 *)*v5 != v5 + 2 )
              _libc_free(*v5);
            j_j___libc_free_0(v6, 48);
          }
        }
      }
      v2 += 2;
    }
    while ( v3 != v2 );
  }
  return j___libc_free_0(*(_QWORD *)(a1 + 8));
}

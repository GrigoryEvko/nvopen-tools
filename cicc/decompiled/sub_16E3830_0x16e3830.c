// Function: sub_16E3830
// Address: 0x16e3830
//
__int64 __fastcall sub_16E3830(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // r13
  _QWORD *v3; // r12
  _QWORD *v4; // rbx

  result = *(unsigned int *)(a1 + 96);
  if ( !(_DWORD)result )
  {
    v2 = *(_QWORD *)(a1 + 264);
    if ( v2 )
    {
      result = *(_QWORD *)(v2 + 8);
      if ( *(_DWORD *)(result + 32) == 4 )
      {
        v3 = *(_QWORD **)(v2 + 48);
        v4 = &v3[4 * *(unsigned int *)(v2 + 56)];
        while ( v3 != v4 )
        {
          while ( 1 )
          {
            v4 -= 4;
            result = (__int64)(v4 + 2);
            if ( (_QWORD *)*v4 == v4 + 2 )
              break;
            result = j_j___libc_free_0(*v4, v4[2] + 1LL);
            if ( v3 == v4 )
              goto LABEL_8;
          }
        }
LABEL_8:
        *(_DWORD *)(v2 + 56) = 0;
      }
    }
  }
  return result;
}

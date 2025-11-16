// Function: sub_27EC8E0
// Address: 0x27ec8e0
//
unsigned __int8 __fastcall sub_27EC8E0(__int64 a1, unsigned __int8 *a2, __m128i a3, __int64 a4, __int64 a5)
{
  unsigned __int8 result; // al
  unsigned __int8 **v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdi
  unsigned __int8 **v12; // rax

  result = *a2;
  if ( *a2 == 62 || result == 61 )
  {
    result = sub_D48480(**(_QWORD **)a1, *((_QWORD *)a2 - 4), *(_QWORD *)a1, a5);
    if ( result )
    {
      v11 = *(_QWORD *)(a1 + 8);
      if ( *(_BYTE *)(v11 + 28) )
      {
        v12 = *(unsigned __int8 ***)(v11 + 8);
        v8 = *(unsigned int *)(v11 + 20);
        v7 = &v12[v8];
        if ( v12 != v7 )
        {
          while ( a2 != *v12 )
          {
            if ( v7 == ++v12 )
              goto LABEL_12;
          }
          return sub_FD98A0(*(__int64 ***)(a1 + 16), a2, a3);
        }
LABEL_12:
        if ( (unsigned int)v8 < *(_DWORD *)(v11 + 16) )
        {
          *(_DWORD *)(v11 + 20) = v8 + 1;
          *v7 = a2;
          ++*(_QWORD *)v11;
          return sub_FD98A0(*(__int64 ***)(a1 + 16), a2, a3);
        }
      }
      sub_C8CC70(v11, (__int64)a2, (__int64)v7, v8, v9, v10);
      return sub_FD98A0(*(__int64 ***)(a1 + 16), a2, a3);
    }
  }
  return result;
}

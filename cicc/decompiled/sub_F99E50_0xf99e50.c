// Function: sub_F99E50
// Address: 0xf99e50
//
unsigned __int64 __fastcall sub_F99E50(__int64 *a1)
{
  __int64 v2; // r14
  unsigned __int64 result; // rax
  __int64 v4; // rbx
  unsigned __int64 v5; // rdi
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // r15
  __int64 v11; // rax

  v2 = *a1;
  result = a1[1];
  *((_BYTE *)a1 + 64) = 0;
  *((_DWORD *)a1 + 6) = 0;
  v4 = v2 + 8 * result;
  if ( v4 != v2 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)v2 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v5 == *(_QWORD *)v2 + 48LL )
      {
        v7 = 0;
      }
      else
      {
        if ( !v5 )
          BUG();
        v6 = *(unsigned __int8 *)(v5 - 24);
        v7 = v5 - 24;
        if ( (unsigned int)(v6 - 30) >= 0xB )
          v7 = 0;
      }
      result = sub_B46BC0(v7, 0);
      v10 = result;
      if ( !result )
        break;
      v11 = *((unsigned int *)a1 + 6);
      if ( v11 + 1 > (unsigned __int64)*((unsigned int *)a1 + 7) )
      {
        sub_C8D5F0((__int64)(a1 + 2), a1 + 4, v11 + 1, 8u, v8, v9);
        v11 = *((unsigned int *)a1 + 6);
      }
      v2 += 8;
      *(_QWORD *)(a1[2] + 8 * v11) = v10;
      result = (unsigned int)(*((_DWORD *)a1 + 6) + 1);
      *((_DWORD *)a1 + 6) = result;
      if ( v4 == v2 )
      {
        if ( (_DWORD)result )
          return result;
        break;
      }
    }
  }
  *((_BYTE *)a1 + 64) = 1;
  return result;
}

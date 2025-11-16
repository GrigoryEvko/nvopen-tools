// Function: sub_15FAA20
// Address: 0x15faa20
//
__int64 __fastcall sub_15FAA20(unsigned __int8 *a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  int v4; // r13d
  unsigned int v5; // r15d
  int v6; // r8d
  unsigned int v7; // r15d
  _QWORD *v8; // r14
  __int64 v9; // rax
  int v10; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  result = (unsigned int)a1[16] - 11;
  v4 = v2;
  if ( (unsigned int)result > 1 )
  {
    if ( (_DWORD)v2 )
    {
      v7 = 0;
      do
      {
        LODWORD(v8) = -1;
        v9 = sub_15A0A60((__int64)a1, v7);
        if ( *(_BYTE *)(v9 + 16) != 9 )
        {
          v8 = *(_QWORD **)(v9 + 24);
          if ( *(_DWORD *)(v9 + 32) > 0x40u )
            v8 = (_QWORD *)*v8;
        }
        result = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
        {
          sub_16CD150(a2, a2 + 16, 0, 4);
          result = *(unsigned int *)(a2 + 8);
        }
        ++v7;
        *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = (_DWORD)v8;
        ++*(_DWORD *)(a2 + 8);
      }
      while ( v4 != v7 );
    }
  }
  else
  {
    v5 = 0;
    if ( (_DWORD)v2 )
    {
      do
      {
        v6 = sub_1595A50((__int64)a1, v5);
        result = *(unsigned int *)(a2 + 8);
        if ( (unsigned int)result >= *(_DWORD *)(a2 + 12) )
        {
          v10 = v6;
          sub_16CD150(a2, a2 + 16, 0, 4);
          result = *(unsigned int *)(a2 + 8);
          v6 = v10;
        }
        ++v5;
        *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v6;
        ++*(_DWORD *)(a2 + 8);
      }
      while ( v4 != v5 );
    }
  }
  return result;
}

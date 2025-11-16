// Function: sub_899910
// Address: 0x899910
//
__int64 __fastcall sub_899910(__int64 a1, __int64 a2, FILE *a3)
{
  __int64 result; // rax
  _QWORD *k; // rbx
  __int64 v7; // r14
  __int64 v8; // rdi
  _QWORD *j; // rbx
  __int64 v10; // rdx
  __int64 v11; // r8
  char v12; // dl
  _QWORD *i; // rbx

  *(_BYTE *)(a2 + 160) |= 1u;
  sub_899850(a1, a3);
  *(_QWORD *)(a2 + 144) = 0;
  result = *(unsigned __int8 *)(a1 + 80);
  if ( (_BYTE)result == 20 )
  {
    for ( i = *(_QWORD **)(a2 + 168); i; i = (_QWORD *)*i )
    {
      result = i[2];
      if ( *(int *)(result + 24) > 0 )
        result = sub_686B60(7u, 0x315u, a3, a1, i[3]);
    }
  }
  else if ( (_BYTE)result == 21 )
  {
    for ( j = *(_QWORD **)(a2 + 184); j; j = (_QWORD *)*j )
    {
      result = sub_892240(j[1]);
      v11 = *(_QWORD *)(result + 24);
      v12 = *(_BYTE *)(v11 + 80);
      if ( v12 == 9 || v12 == 7 )
      {
        v10 = *(_QWORD *)(v11 + 88);
      }
      else
      {
        if ( v12 != 21 )
          BUG();
        v10 = *(_QWORD *)(*(_QWORD *)(v11 + 88) + 192LL);
      }
      if ( (*(_BYTE *)(v10 + 170) & 0x40) == 0 )
      {
        result = *(unsigned int *)(*(_QWORD *)(result + 16) + 24LL);
        if ( (int)result > 0 )
          result = sub_686B60(7u, 0x315u, a3, a1, v11);
      }
    }
  }
  else
  {
    for ( k = *(_QWORD **)(a2 + 168); k; k = (_QWORD *)*k )
    {
      while ( 1 )
      {
        v7 = k[1];
        v8 = *(_QWORD *)(v7 + 88);
        if ( (unsigned __int8)(*(_BYTE *)(v7 + 80) - 4) > 1u )
          break;
        result = *(_BYTE *)(v8 + 177) & 0x30;
        if ( (_BYTE)result != 48 )
          break;
        k = (_QWORD *)*k;
        if ( !k )
          return result;
      }
      result = sub_8D23B0(v8);
      if ( !(_DWORD)result )
      {
        result = *(_QWORD *)(v7 + 88);
        if ( (*(_BYTE *)(result + 178) & 1) == 0 )
          result = sub_686B60(7u, 0x315u, a3, a1, v7);
      }
    }
  }
  return result;
}

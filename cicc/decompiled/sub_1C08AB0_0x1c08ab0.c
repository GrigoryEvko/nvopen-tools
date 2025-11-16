// Function: sub_1C08AB0
// Address: 0x1c08ab0
//
__int64 __fastcall sub_1C08AB0(__int64 a1, __int64 *a2, __int64 *a3, _DWORD *a4)
{
  unsigned int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rsi
  __int64 v11; // r11
  __int64 v12; // r10
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rcx

  if ( *((_DWORD *)a2 + 8) != *((_DWORD *)a3 + 8) )
    return 0;
  *a4 = 0;
  v7 = *((_DWORD *)a2 + 8);
  v8 = *a2;
  v9 = *a3;
  if ( v7 )
  {
    v10 = 0;
    v11 = 8LL * v7;
    v12 = v7 - 1;
    while ( 2 )
    {
      v13 = *(_QWORD **)(a2[3] + v10);
      if ( v8 == *v13 )
      {
        *a4 |= 1u;
      }
      else if ( v9 == *v13 )
      {
        *a4 |= 2u;
      }
      v14 = (_QWORD *)a3[3];
      v15 = (__int64)&v14[v12 + 1];
      while ( (_QWORD *)*v14 != v13 )
      {
        if ( (_QWORD *)v15 == ++v14 )
          return 0;
      }
      v10 += 8;
      if ( v10 != v11 )
        continue;
      break;
    }
  }
  return 1;
}

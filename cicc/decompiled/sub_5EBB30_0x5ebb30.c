// Function: sub_5EBB30
// Address: 0x5ebb30
//
_QWORD *__fastcall sub_5EBB30(__int64 a1, _QWORD *a2, _QWORD *a3, char a4)
{
  _QWORD *v4; // r13
  __int64 v6; // rbx
  _QWORD *v7; // rax
  _QWORD *v8; // r14
  _QWORD *v9; // r15
  _QWORD *v10; // rax
  __int64 *v12; // rax
  __int64 *v13; // rdx
  _QWORD *v14; // [rsp+8h] [rbp-48h]

  v4 = a2;
  v6 = sub_724FE0();
  v7 = sub_5EBAE0(a1, 0);
  v8 = v7;
  if ( a2 )
  {
    if ( *a3 )
    {
      if ( (_QWORD *)*a3 == a2 )
      {
        MEMORY[0] = v7;
        BUG();
      }
      v14 = sub_5EBAE0(a2[2], 0);
      v9 = v14;
      v14[1] = 0;
      while ( 1 )
      {
        v4 = (_QWORD *)*v4;
        if ( (_QWORD *)*a3 == v4 )
          break;
        v10 = sub_5EBAE0(v4[2], 0);
        v10[1] = v9;
        *v9 = v10;
        v9 = v10;
      }
      *v9 = v8;
      v8[1] = v9;
      *(_QWORD *)(v6 + 8) = v14;
      *(_QWORD *)(v6 + 16) = v8;
    }
    else
    {
      v7[1] = a3;
      *a3 = v7;
      *(_QWORD *)(v6 + 8) = a2;
      *(_QWORD *)(v6 + 16) = v7;
    }
  }
  else
  {
    *(_BYTE *)(v6 + 24) |= 1u;
    *(_QWORD *)(v6 + 8) = v7;
    *(_QWORD *)(v6 + 16) = v7;
  }
  *(_BYTE *)(v6 + 25) = a4;
  if ( (*(_BYTE *)(a1 + 96) & 2) != 0 )
  {
    v12 = *(__int64 **)(a1 + 112);
    if ( v12 )
    {
      do
      {
        v13 = v12;
        v12 = (__int64 *)*v12;
      }
      while ( v12 );
      *v13 = v6;
    }
    else
    {
      *(_QWORD *)(a1 + 112) = v6;
    }
  }
  else
  {
    *(_BYTE *)(v6 + 24) |= 2u;
    *(_QWORD *)(a1 + 112) = v6;
    return *(_QWORD **)(v6 + 8);
  }
  return v8;
}

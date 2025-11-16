// Function: sub_393CD60
// Address: 0x393cd60
//
__int64 __fastcall sub_393CD60(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  __int64 result; // rax
  _QWORD *v6; // r13
  unsigned __int64 v7; // rbx
  _QWORD *v8; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  char v15; // di
  _QWORD *v16; // [rsp+8h] [rbp-38h]
  _QWORD *v17; // [rsp+8h] [rbp-38h]

  ++*(_DWORD *)(a1 + 124);
  v3 = *(_QWORD *)(a2 + 24);
  if ( *(_QWORD *)(a1 + 112) < v3 )
    *(_QWORD *)(a1 + 112) = v3;
  v4 = *(_QWORD *)(a2 + 56);
  result = a2 + 40;
  v6 = (_QWORD *)(a1 + 8);
  if ( v4 != a2 + 40 )
  {
    do
    {
      v7 = *(_QWORD *)(v4 + 40);
      *(_QWORD *)(a1 + 96) += v7;
      if ( v7 > *(_QWORD *)(a1 + 104) )
        *(_QWORD *)(a1 + 104) = v7;
      v8 = *(_QWORD **)(a1 + 16);
      ++*(_DWORD *)(a1 + 120);
      v9 = a1 + 8;
      if ( !v8 )
        goto LABEL_13;
      do
      {
        while ( 1 )
        {
          v10 = v8[2];
          v11 = v8[3];
          if ( v7 >= v8[4] )
            break;
          v8 = (_QWORD *)v8[3];
          if ( !v11 )
            goto LABEL_11;
        }
        v9 = (unsigned __int64)v8;
        v8 = (_QWORD *)v8[2];
      }
      while ( v10 );
LABEL_11:
      if ( v6 == (_QWORD *)v9 || v7 > *(_QWORD *)(v9 + 32) )
      {
LABEL_13:
        v16 = (_QWORD *)v9;
        v12 = sub_22077B0(0x30u);
        *(_QWORD *)(v12 + 32) = v7;
        v9 = v12;
        *(_DWORD *)(v12 + 40) = 0;
        v13 = sub_EFBD70((_QWORD *)a1, v16, (unsigned __int64 *)(v12 + 32));
        if ( v14 )
        {
          v15 = v6 == v14 || v13 || v7 > v14[4];
          sub_220F040(v15, v9, v14, (_QWORD *)(a1 + 8));
          ++*(_QWORD *)(a1 + 40);
        }
        else
        {
          v17 = v13;
          j_j___libc_free_0(v9);
          v9 = (unsigned __int64)v17;
        }
      }
      ++*(_DWORD *)(v9 + 40);
      result = sub_220EF30(v4);
      v4 = result;
    }
    while ( a2 + 40 != result );
  }
  return result;
}

// Function: sub_1DB4840
// Address: 0x1db4840
//
__int64 __fastcall sub_1DB4840(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r13
  _QWORD *v5; // rbx
  __int64 v6; // rdx
  _QWORD *v7; // rcx
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned int v13; // edx
  __int64 v14; // rcx

  v3 = a3;
  v4 = a2;
  if ( *(_DWORD *)a2 < *(_DWORD *)a3 )
  {
    v4 = a3;
    v3 = a2;
    *(_QWORD *)(a2 + 8) = *(_QWORD *)(a3 + 8);
  }
  v5 = *(_QWORD **)a1;
  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(_QWORD **)a1;
  while ( 1 )
  {
    v8 = &v7[3 * v6];
    if ( v5 == v8 )
      break;
    while ( 1 )
    {
      v9 = v5 + 3;
      if ( v5[2] == v4 )
        break;
      v5 += 3;
      if ( v9 == v8 )
        goto LABEL_7;
    }
    if ( v5 != v7 && *(v5 - 1) == v3 && *v5 == *(v5 - 2) )
    {
      *(v5 - 2) = v5[1];
      v13 = *(_DWORD *)(a1 + 8);
      v14 = *(_QWORD *)a1 + 24LL * v13;
      if ( v9 != (_QWORD *)v14 )
      {
        memmove(v5, v5 + 3, v14 - (_QWORD)v9);
        v13 = *(_DWORD *)(a1 + 8);
      }
      v9 = v5;
      v5 -= 3;
      *(_DWORD *)(a1 + 8) = v13 - 1;
    }
    v5[2] = v3;
    v6 = *(unsigned int *)(a1 + 8);
    v7 = *(_QWORD **)a1;
    if ( v9 != (_QWORD *)(*(_QWORD *)a1 + 24 * v6) && v5[1] == *v9 && v9[2] == v3 )
    {
      v5[1] = v9[1];
      v7 = *(_QWORD **)a1;
      v11 = *(unsigned int *)(a1 + 8);
      v12 = *(_QWORD *)a1 + 24 * v11;
      if ( (_QWORD *)v12 != v9 + 3 )
      {
        memmove(v9, v9 + 3, v12 - (_QWORD)(v9 + 3));
        LODWORD(v11) = *(_DWORD *)(a1 + 8);
        v7 = *(_QWORD **)a1;
      }
      v6 = (unsigned int)(v11 - 1);
      v9 = v5 + 3;
      *(_DWORD *)(a1 + 8) = v6;
    }
    v5 = v9;
  }
LABEL_7:
  sub_1DB4220(a1, v4);
  return v3;
}

// Function: sub_937570
// Address: 0x937570
//
__int64 __fastcall sub_937570(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rax
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rcx
  __int64 v6; // rax
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r14
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rax
  int v17; // r15d
  unsigned __int64 v18; // rcx
  __int64 v19; // rax
  int v20; // r15d

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL);
  v3 = *(unsigned int *)(a2 + 8);
  if ( v3 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v3 + 1, 4);
    v3 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v3) = v2;
  v4 = HIDWORD(v2);
  v5 = *(unsigned int *)(a2 + 12);
  v6 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = v6;
  if ( v6 + 1 > v5 )
  {
    sub_C8D5F0(a2, a2 + 16, v6 + 1, 4);
    v6 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v6) = v4;
  result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = result;
  v8 = *(_QWORD *)(a1 + 16);
  v9 = v8 + 40;
  v10 = v8 + 8 * (5LL * *(unsigned int *)(a1 + 8) + 5);
  if ( v10 != v8 + 40 )
  {
    do
    {
      v11 = *(_QWORD *)(v9 + 24);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        sub_C8D5F0(a2, a2 + 16, result + 1, 4);
        result = *(unsigned int *)(a2 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v11;
      v12 = HIDWORD(v11);
      v13 = *(unsigned int *)(a2 + 12);
      v14 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v14;
      if ( v14 + 1 > v13 )
      {
        sub_C8D5F0(a2, a2 + 16, v14 + 1, 4);
        v14 = *(unsigned int *)(a2 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a2 + 4 * v14) = v12;
      v15 = *(unsigned int *)(a2 + 12);
      v16 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v16;
      v17 = *(unsigned __int8 *)(v9 + 32);
      if ( v16 + 1 > v15 )
      {
        sub_C8D5F0(a2, a2 + 16, v16 + 1, 4);
        v16 = *(unsigned int *)(a2 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a2 + 4 * v16) = v17;
      v18 = *(unsigned int *)(a2 + 12);
      v19 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v19;
      v20 = *(unsigned __int8 *)(v9 + 33);
      if ( v19 + 1 > v18 )
      {
        sub_C8D5F0(a2, a2 + 16, v19 + 1, 4);
        v19 = *(unsigned int *)(a2 + 8);
      }
      v9 += 40;
      *(_DWORD *)(*(_QWORD *)a2 + 4 * v19) = v20;
      result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = result;
    }
    while ( v9 != v10 );
  }
  return result;
}

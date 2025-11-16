// Function: sub_15230E0
// Address: 0x15230e0
//
__int64 __fastcall sub_15230E0(__int64 *a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r15
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r12
  unsigned __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // [rsp+8h] [rbp-38h]

  v4 = a2;
  v5 = a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((v5 - *a1) >> 3);
  if ( a2 >= (unsigned int)v7 )
  {
    v11 = a2 + 1;
    if ( v11 > v7 )
    {
      sub_14EF7A0((__int64)a1, v11 - v7);
      v6 = *a1;
    }
    else if ( v11 < v7 )
    {
      v23 = v6 + 24 * v11;
      if ( v5 != v23 )
      {
        v12 = v6 + 24 * v11;
        do
        {
          v13 = *(_QWORD *)(v12 + 16);
          if ( v13 != 0 && v13 != -8 && v13 != -16 )
            sub_1649B30(v12);
          v12 += 24;
        }
        while ( v5 != v12 );
        v6 = *a1;
        a1[1] = v23;
      }
    }
  }
  v8 = 24 * v4;
  v9 = *(_QWORD *)(v6 + v8 + 16);
  if ( v9 )
  {
    if ( a3 != *(_QWORD *)v9 )
      sub_16BD130("Type mismatch in constant table!", 1);
  }
  else
  {
    v14 = sub_1648A60(24, 1);
    v9 = v14;
    if ( v14 )
    {
      v15 = a1[6];
      sub_1648CB0(v14, a3, 5);
      *(_DWORD *)(v9 + 20) = *(_DWORD *)(v9 + 20) & 0xF0000000 | 1;
      *(_WORD *)(v9 + 18) = 56;
      v16 = sub_1643350(v15);
      v17 = sub_1599EF0(v16);
      if ( *(_QWORD *)(v9 - 24) )
      {
        v18 = *(_QWORD *)(v9 - 16);
        v19 = *(_QWORD *)(v9 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v19 = v18;
        if ( v18 )
          *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
      }
      *(_QWORD *)(v9 - 24) = v17;
      if ( v17 )
      {
        v20 = *(_QWORD *)(v17 + 8);
        *(_QWORD *)(v9 - 16) = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = (v9 - 16) | *(_QWORD *)(v20 + 16) & 3LL;
        *(_QWORD *)(v9 - 8) = (v17 + 8) | *(_QWORD *)(v9 - 8) & 3LL;
        *(_QWORD *)(v17 + 8) = v9 - 24;
      }
    }
    v21 = *a1 + v8;
    v22 = *(_QWORD *)(v21 + 16);
    if ( v9 != v22 )
    {
      if ( v22 != -8 && v22 != 0 && v22 != -16 )
        sub_1649B30(v21);
      *(_QWORD *)(v21 + 16) = v9;
      if ( v9 && v9 != -8 && v9 != -16 )
        sub_164C220(v21);
    }
  }
  return v9;
}

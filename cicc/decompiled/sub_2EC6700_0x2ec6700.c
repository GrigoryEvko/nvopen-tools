// Function: sub_2EC6700
// Address: 0x2ec6700
//
void __fastcall sub_2EC6700(_QWORD *a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  __int64 v4; // rdi
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // r14
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned __int64 *v12; // rbx
  _BYTE *v13; // r14
  __int64 v14; // rdx
  _BYTE *v15; // rax
  unsigned __int64 *v16; // r8
  unsigned __int64 *v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  _BYTE *v21; // rax
  unsigned __int64 *v22; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v23; // [rsp+8h] [rbp-38h]

  v2 = (unsigned __int64 *)a1[421];
  if ( v2 )
  {
    v3 = (unsigned __int64 *)a1[114];
    v4 = a1[113];
    if ( v2 != v3 )
    {
      v5 = v2;
      if ( (*(_BYTE *)v2 & 4) == 0 && (*((_BYTE *)v2 + 44) & 8) != 0 )
      {
        do
          v5 = (unsigned __int64 *)v5[1];
        while ( (*((_BYTE *)v5 + 44) & 8) != 0 );
      }
      v6 = (unsigned __int64 *)v5[1];
      if ( v2 == v6 || v3 == v6 )
      {
        v3 = v2;
      }
      else
      {
        sub_2E310C0((__int64 *)(v4 + 40), (__int64 *)(v4 + 40), (__int64)v2, v5[1]);
        if ( v6 != v2 )
        {
          v7 = *v6 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((*v2 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v6;
          *v6 = *v6 & 7 | *v2 & 0xFFFFFFFFFFFFFFF8LL;
          v8 = *v3;
          *(_QWORD *)(v7 + 8) = v3;
          v8 &= 0xFFFFFFFFFFFFFFF8LL;
          *v2 = v8 | *v2 & 7;
          *(_QWORD *)(v8 + 8) = v2;
          *v3 = v7 | *v3 & 7;
        }
        v3 = (unsigned __int64 *)a1[421];
      }
    }
    a1[114] = v3;
  }
  v9 = a1[419];
  v10 = a1[418];
  while ( v10 != v9 )
  {
    v11 = a1[114];
    v12 = *(unsigned __int64 **)(v9 - 16);
    v9 -= 16;
    v13 = *(_BYTE **)(v9 + 8);
    if ( v12 == (unsigned __int64 *)v11 )
    {
      if ( !v12 )
        BUG();
      if ( (*(_BYTE *)v12 & 4) == 0 && (*((_BYTE *)v12 + 44) & 8) != 0 )
      {
        do
          v11 = *(_QWORD *)(v11 + 8);
        while ( (*(_BYTE *)(v11 + 44) & 8) != 0 );
      }
      a1[114] = *(_QWORD *)(v11 + 8);
    }
    v14 = a1[113];
    if ( !v13 )
      BUG();
    v15 = v13;
    if ( (*v13 & 4) == 0 && (v13[44] & 8) != 0 )
    {
      do
        v15 = (_BYTE *)*((_QWORD *)v15 + 1);
      while ( (v15[44] & 8) != 0 );
    }
    v16 = (unsigned __int64 *)*((_QWORD *)v15 + 1);
    if ( v12 != v16 )
    {
      if ( !v12 )
        BUG();
      v17 = v12;
      if ( (*(_BYTE *)v12 & 4) == 0 && (*((_BYTE *)v12 + 44) & 8) != 0 )
      {
        do
          v17 = (unsigned __int64 *)v17[1];
        while ( (*((_BYTE *)v17 + 44) & 8) != 0 );
      }
      v18 = v17[1];
      if ( v12 != (unsigned __int64 *)v18 )
      {
        v23 = v16;
        if ( (unsigned __int64 *)v18 != v16 )
        {
          v22 = (unsigned __int64 *)v18;
          sub_2E310C0((__int64 *)(v14 + 40), (__int64 *)(v14 + 40), (__int64)v12, v18);
          if ( v22 != v12 )
          {
            v19 = *v22 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v22;
            *v22 = *v22 & 7 | *v12 & 0xFFFFFFFFFFFFFFF8LL;
            v20 = *v23;
            *(_QWORD *)(v19 + 8) = v23;
            v20 &= 0xFFFFFFFFFFFFFFF8LL;
            *v12 = v20 | *v12 & 7;
            *(_QWORD *)(v20 + 8) = v12;
            *v23 = v19 | *v23 & 7;
          }
          v14 = a1[113];
        }
      }
    }
    v21 = (_BYTE *)a1[115];
    if ( v21 != (_BYTE *)(v14 + 48) && v21 == v13 )
      a1[115] = v12;
  }
}

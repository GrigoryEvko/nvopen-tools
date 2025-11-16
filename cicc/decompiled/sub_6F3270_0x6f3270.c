// Function: sub_6F3270
// Address: 0x6f3270
//
__int64 __fastcall sub_6F3270(unsigned __int64 a1, unsigned __int64 a2, int *a3)
{
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rbx
  char v6; // al
  char v7; // al
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 *v10; // r15
  int v11; // ebx
  int v12; // edx
  __int64 result; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // eax

  v4 = a2;
  v5 = a1;
  v6 = *(_BYTE *)(a1 + 80);
  if ( v6 == 16 )
  {
    v5 = **(_QWORD **)(a1 + 88);
    v6 = *(_BYTE *)(v5 + 80);
  }
  if ( v6 == 24 )
  {
    v5 = *(_QWORD *)(v5 + 88);
    v6 = *(_BYTE *)(v5 + 80);
  }
  if ( (unsigned __int8)(v6 - 10) <= 1u )
  {
    v16 = *(_QWORD *)(v5 + 88);
    if ( (*(_BYTE *)(v16 + 194) & 0x40) != 0 )
    {
      do
        v16 = *(_QWORD *)(v16 + 232);
      while ( (*(_BYTE *)(v16 + 194) & 0x40) != 0 );
      v5 = *(_QWORD *)v16;
    }
  }
  else if ( v6 == 20 )
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v5 + 88) + 176LL);
    if ( (*(_BYTE *)(v15 + 194) & 0x40) != 0 )
    {
      do
        v15 = *(_QWORD *)(v15 + 232);
      while ( (*(_BYTE *)(v15 + 194) & 0x40) != 0 );
      v5 = **(_QWORD **)(v15 + 248);
    }
  }
  v7 = *(_BYTE *)(a2 + 80);
  if ( v7 == 16 )
  {
    v4 = **(_QWORD **)(a2 + 88);
    v7 = *(_BYTE *)(v4 + 80);
  }
  if ( v7 == 24 )
  {
    v4 = *(_QWORD *)(v4 + 88);
    v7 = *(_BYTE *)(v4 + 80);
  }
  if ( (unsigned __int8)(v7 - 10) <= 1u )
  {
    v17 = *(_QWORD *)(v4 + 88);
    if ( (*(_BYTE *)(v17 + 194) & 0x40) != 0 )
    {
      do
        v17 = *(_QWORD *)(v17 + 232);
      while ( (*(_BYTE *)(v17 + 194) & 0x40) != 0 );
      v4 = *(_QWORD *)v17;
    }
  }
  else if ( v7 == 20 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v4 + 88) + 176LL);
    if ( (*(_BYTE *)(v14 + 194) & 0x40) != 0 )
    {
      do
        v14 = *(_QWORD *)(v14 + 232);
      while ( (*(_BYTE *)(v14 + 194) & 0x40) != 0 );
      v4 = **(_QWORD **)(v14 + 248);
    }
  }
  v8 = sub_6F27E0(v5);
  v9 = sub_6F27E0(v4);
  v10 = (__int64 *)v9;
  if ( v8 == 1 )
  {
    if ( v9 != 1 )
    {
LABEL_20:
      v11 = 0;
      result = 0xFFFFFFFFLL;
      goto LABEL_23;
    }
    v11 = 1;
    result = 0;
  }
  else
  {
    if ( v9 != 1 )
    {
      if ( v4 == v5 )
      {
LABEL_21:
        v11 = 0;
LABEL_22:
        result = 0;
        goto LABEL_23;
      }
      if ( (*(_BYTE *)(v9 + 24) & 1) != 0 )
      {
        if ( (*(_BYTE *)(v8 + 24) & 1) == 0 )
        {
          v11 = 0;
          v12 = sub_6E0210((__int64 *)v9, (__int64 *)v8);
          goto LABEL_19;
        }
        goto LABEL_21;
      }
      v18 = sub_6E0210((__int64 *)v8, (__int64 *)v9);
      v11 = v18;
      if ( (*(_BYTE *)(v8 + 24) & 1) != 0 )
      {
        if ( (v18 & 1) == 0 )
          goto LABEL_22;
      }
      else
      {
        v12 = sub_6E0210(v10, (__int64 *)v8);
        if ( (v11 & (v12 ^ 1)) == 0 )
        {
LABEL_19:
          result = v12 & (v11 ^ 1u);
          if ( (v12 & (v11 ^ 1)) == 0 )
            goto LABEL_23;
          goto LABEL_20;
        }
      }
    }
    v11 = 0;
    result = 1;
  }
LABEL_23:
  if ( a3 )
    *a3 = v11;
  return result;
}

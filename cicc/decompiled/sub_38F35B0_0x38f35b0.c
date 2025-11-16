// Function: sub_38F35B0
// Address: 0x38f35b0
//
char __fastcall sub_38F35B0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rdx
  unsigned int v6; // ecx
  char result; // al
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rdi
  const char *v22; // rax
  __int64 v23; // [rsp+8h] [rbp-58h] BYREF
  __int64 v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25; // [rsp+18h] [rbp-48h]
  _QWORD v26[2]; // [rsp+20h] [rbp-40h] BYREF
  char v27; // [rsp+30h] [rbp-30h]
  char v28; // [rsp+31h] [rbp-2Fh]

  v2 = *(_QWORD *)a1;
  v24 = 0;
  v25 = 0;
  v3 = sub_3909460(v2);
  v4 = sub_39092A0(v3);
  result = sub_38F0EE0(*(_QWORD *)a1, &v24, v5, v6);
  if ( result )
  {
    v16 = *(_QWORD *)a1;
    v28 = 1;
    v26[0] = "unexpected token in '.loc' directive";
    v27 = 3;
    return sub_3909CF0(v16, v26, 0, 0, v8, v9);
  }
  switch ( v25 )
  {
    case 11LL:
      if ( *(_QWORD *)v24 == 0x6C625F6369736162LL && *(_WORD *)(v24 + 8) == 25455 && *(_BYTE *)(v24 + 10) == 107 )
      {
        **(_DWORD **)(a1 + 8) |= 2u;
        return result;
      }
LABEL_14:
      v10 = *(_QWORD *)a1;
      goto LABEL_15;
    case 12LL:
      if ( *(_QWORD *)v24 == 0x6575676F6C6F7270LL && *(_DWORD *)(v24 + 8) == 1684956511 )
      {
        **(_DWORD **)(a1 + 8) |= 4u;
        return result;
      }
      goto LABEL_14;
    case 14LL:
      if ( *(_QWORD *)v24 == 0x6575676F6C697065LL && *(_DWORD *)(v24 + 8) == 1734697567 && *(_WORD *)(v24 + 12) == 28265 )
      {
        **(_DWORD **)(a1 + 8) |= 8u;
        return result;
      }
      goto LABEL_14;
  }
  v10 = *(_QWORD *)a1;
  if ( v25 != 7 )
  {
    if ( v25 == 3 )
    {
      if ( *(_WORD *)v24 == 29545 && *(_BYTE *)(v24 + 2) == 97 )
      {
        v11 = sub_3909460(v10);
        v12 = sub_39092A0(v11);
        v13 = *(_QWORD *)a1;
        v26[0] = 0;
        v14 = v12;
        result = sub_38EB6A0(v13, &v23, (__int64)v26);
        if ( !result )
        {
          if ( *(_DWORD *)v23 == 1 )
          {
            v15 = *(_QWORD *)(v23 + 16);
            if ( (int)v15 >= 0 )
            {
              **(_DWORD **)(a1 + 16) = v15;
              return result;
            }
            v28 = 1;
            v21 = *(_QWORD *)a1;
            v22 = "isa number less than zero";
          }
          else
          {
            v28 = 1;
            v21 = *(_QWORD *)a1;
            v22 = "isa number not a constant value";
          }
          goto LABEL_41;
        }
        return result;
      }
    }
    else if ( v25 == 13
           && *(_QWORD *)v24 == 0x696D697263736964LL
           && *(_DWORD *)(v24 + 8) == 1869898094
           && *(_BYTE *)(v24 + 12) == 114 )
    {
      return sub_38EB9C0(v10, *(_QWORD **)(a1 + 24));
    }
LABEL_15:
    v28 = 1;
    v26[0] = "unknown sub-directive in '.loc' directive";
    v27 = 3;
    return sub_3909790(v10, v4, v26, 0, 0);
  }
  if ( *(_DWORD *)v24 != 1935635305 || *(_WORD *)(v24 + 4) != 28020 || *(_BYTE *)(v24 + 6) != 116 )
    goto LABEL_15;
  v17 = sub_3909460(v10);
  v18 = sub_39092A0(v17);
  v19 = *(_QWORD *)a1;
  v26[0] = 0;
  v14 = v18;
  result = sub_38EB6A0(v19, &v23, (__int64)v26);
  if ( !result )
  {
    if ( *(_DWORD *)v23 == 1 )
    {
      v20 = *(_QWORD *)(v23 + 16);
      if ( !(_DWORD)v20 )
      {
        **(_DWORD **)(a1 + 8) &= ~1u;
        return result;
      }
      if ( (_DWORD)v20 == 1 )
      {
        **(_DWORD **)(a1 + 8) |= 1u;
        return result;
      }
      v28 = 1;
      v21 = *(_QWORD *)a1;
      v22 = "is_stmt value not 0 or 1";
    }
    else
    {
      v28 = 1;
      v21 = *(_QWORD *)a1;
      v22 = "is_stmt value not the constant value of 0 or 1";
    }
LABEL_41:
    v26[0] = v22;
    v27 = 3;
    return sub_3909790(v21, v14, v26, 0, 0);
  }
  return result;
}

// Function: sub_EB84F0
// Address: 0xeb84f0
//
char __fastcall sub_EB84F0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r12
  char result; // al
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rdi
  const char *v19; // rax
  __int64 v20; // [rsp+8h] [rbp-68h] BYREF
  __int64 v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 v22; // [rsp+18h] [rbp-58h]
  _QWORD v23[4]; // [rsp+20h] [rbp-50h] BYREF
  char v24; // [rsp+40h] [rbp-30h]
  char v25; // [rsp+41h] [rbp-2Fh]

  v2 = *(_QWORD *)a1;
  v21 = 0;
  v22 = 0;
  v3 = sub_ECD7B0(v2);
  v4 = sub_ECD6A0(v3);
  result = sub_EB61F0(*(_QWORD *)a1, &v21);
  if ( result )
  {
    v7 = *(_QWORD *)a1;
    v25 = 1;
    v23[0] = "unexpected token in '.loc' directive";
    v24 = 3;
    return sub_ECE0E0(v7, v23, 0, 0);
  }
  switch ( v22 )
  {
    case 11LL:
      if ( *(_QWORD *)v21 == 0x6C625F6369736162LL && *(_WORD *)(v21 + 8) == 25455 && *(_BYTE *)(v21 + 10) == 107 )
      {
        **(_DWORD **)(a1 + 8) |= 2u;
        return result;
      }
LABEL_6:
      v6 = *(_QWORD *)a1;
      goto LABEL_7;
    case 12LL:
      if ( *(_QWORD *)v21 == 0x6575676F6C6F7270LL && *(_DWORD *)(v21 + 8) == 1684956511 )
      {
        **(_DWORD **)(a1 + 8) |= 4u;
        return result;
      }
      goto LABEL_6;
    case 14LL:
      if ( *(_QWORD *)v21 == 0x6575676F6C697065LL && *(_DWORD *)(v21 + 8) == 1734697567 && *(_WORD *)(v21 + 12) == 28265 )
      {
        **(_DWORD **)(a1 + 8) |= 8u;
        return result;
      }
      goto LABEL_6;
    case 7LL:
      v6 = *(_QWORD *)a1;
      if ( *(_DWORD *)v21 != 1935635305 || *(_WORD *)(v21 + 4) != 28020 || *(_BYTE *)(v21 + 6) != 116 )
        goto LABEL_7;
      v8 = sub_ECD7B0(v6);
      v9 = sub_ECD6A0(v8);
      v10 = *(_QWORD *)a1;
      v23[0] = 0;
      v11 = v9;
      result = sub_EAC4D0(v10, &v20, (__int64)v23);
      if ( !result )
      {
        if ( *(_BYTE *)v20 == 1 )
        {
          v12 = *(_QWORD *)(v20 + 16);
          if ( !(_DWORD)v12 )
          {
            **(_DWORD **)(a1 + 8) &= ~1u;
            return result;
          }
          if ( (_DWORD)v12 == 1 )
          {
            **(_DWORD **)(a1 + 8) |= 1u;
            return result;
          }
          v25 = 1;
          v18 = *(_QWORD *)a1;
          v19 = "is_stmt value not 0 or 1";
        }
        else
        {
          v25 = 1;
          v18 = *(_QWORD *)a1;
          v19 = "is_stmt value not the constant value of 0 or 1";
        }
LABEL_42:
        v23[0] = v19;
        v24 = 3;
        return sub_ECDA70(v18, v11, v23, 0, 0);
      }
      break;
    case 3LL:
      if ( *(_WORD *)v21 != 29545 || (v13 = 0, *(_BYTE *)(v21 + 2) != 97) )
        v13 = 1;
      v6 = *(_QWORD *)a1;
      if ( v13 )
      {
LABEL_7:
        v25 = 1;
        v23[0] = "unknown sub-directive in '.loc' directive";
        v24 = 3;
        return sub_ECDA70(v6, v4, v23, 0, 0);
      }
      v14 = sub_ECD7B0(v6);
      v15 = sub_ECD6A0(v14);
      v16 = *(_QWORD *)a1;
      v23[0] = 0;
      v11 = v15;
      result = sub_EAC4D0(v16, &v20, (__int64)v23);
      if ( !result )
      {
        if ( *(_BYTE *)v20 == 1 )
        {
          v17 = *(_QWORD *)(v20 + 16);
          if ( (int)v17 >= 0 )
          {
            **(_DWORD **)(a1 + 16) = v17;
            return result;
          }
          v25 = 1;
          v18 = *(_QWORD *)a1;
          v19 = "isa number less than zero";
        }
        else
        {
          v25 = 1;
          v18 = *(_QWORD *)a1;
          v19 = "isa number not a constant value";
        }
        goto LABEL_42;
      }
      break;
    default:
      v6 = *(_QWORD *)a1;
      if ( v22 == 13
        && *(_QWORD *)v21 == 0x696D697263736964LL
        && *(_DWORD *)(v21 + 8) == 1869898094
        && *(_BYTE *)(v21 + 12) == 114 )
      {
        return sub_EAC8B0(v6, *(_QWORD **)(a1 + 24));
      }
      goto LABEL_7;
  }
  return result;
}

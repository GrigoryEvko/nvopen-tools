// Function: sub_15FC090
// Address: 0x15fc090
//
__int64 __fastcall sub_15FC090(int a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // r15
  char v5; // r14
  unsigned int v6; // r12d
  __int64 v8; // rax
  int v9; // r9d
  int v10; // edx
  unsigned int v11; // ecx
  int v12; // esi
  int v13; // eax
  unsigned int v14; // edx
  char v15; // cl
  __int64 v16; // rax
  unsigned __int8 v17; // si
  __int64 v18; // rcx
  unsigned int v19; // eax
  unsigned int v20; // edx
  unsigned int v21; // eax
  unsigned int v22; // edx
  char v23; // cl
  __int64 v24; // rax
  __int64 v25; // rcx
  int v26; // ebx
  unsigned int v27; // [rsp+0h] [rbp-40h]
  unsigned int v28; // [rsp+0h] [rbp-40h]
  unsigned int v29; // [rsp+8h] [rbp-38h]
  unsigned __int8 v30; // [rsp+Ch] [rbp-34h]
  int v31; // [rsp+Ch] [rbp-34h]
  int v32; // [rsp+Ch] [rbp-34h]
  int v33; // [rsp+Ch] [rbp-34h]

  v3 = a3;
  v4 = *a2;
  v5 = *(_BYTE *)(*a2 + 8LL);
  LOBYTE(a3) = v5 != 12 && v5 != 0;
  v6 = a3;
  if ( (_BYTE)a3 )
  {
    if ( (unsigned __int8)(v5 - 13) <= 1u
      || (unsigned __int8)(*(_BYTE *)(v3 + 8) - 13) <= 1u
      || *(_BYTE *)(v3 + 8) == 12
      || *(_BYTE *)(v3 + 8) == 0 )
    {
      return 0;
    }
    else
    {
      v30 = *(_BYTE *)(v3 + 8);
      v29 = sub_16431D0(*a2);
      LODWORD(v8) = sub_16431D0(v3);
      v9 = 0;
      v10 = v30;
      v11 = v8;
      if ( v5 == 16 )
        v9 = *(_DWORD *)(v4 + 32);
      v12 = 0;
      if ( v30 == 16 )
        v12 = *(_DWORD *)(v3 + 32);
      switch ( a1 )
      {
        case '$':
          if ( v5 == 16 )
          {
            v8 = **(_QWORD **)(v4 + 16);
            v5 = *(_BYTE *)(v8 + 8);
          }
          v6 = 0;
          if ( v5 == 11 )
          {
            if ( v30 == 16 )
            {
              v8 = *(_QWORD *)(v3 + 16);
              v3 = *(_QWORD *)v8;
            }
            v6 = 0;
            if ( *(_BYTE *)(v3 + 8) == 11 )
            {
              LOBYTE(v10) = v9 == v12;
              LOBYTE(v8) = v29 > v11;
              return (unsigned int)v8 & v10;
            }
          }
          return v6;
        case '%':
        case '&':
          if ( v5 == 16 )
          {
            v8 = **(_QWORD **)(v4 + 16);
            v5 = *(_BYTE *)(v8 + 8);
          }
          v6 = 0;
          if ( v5 == 11 )
          {
            if ( v30 == 16 )
            {
              v8 = *(_QWORD *)(v3 + 16);
              v3 = *(_QWORD *)v8;
            }
            v6 = 0;
            if ( *(_BYTE *)(v3 + 8) == 11 )
            {
              LOBYTE(v10) = v9 == v12;
              LOBYTE(v8) = v29 < v11;
              return (unsigned int)v8 & v10;
            }
          }
          return v6;
        case '\'':
        case '(':
          if ( v5 == 16 )
          {
            v8 = **(_QWORD **)(v4 + 16);
            v5 = *(_BYTE *)(v8 + 8);
          }
          if ( (unsigned __int8)(v5 - 1) > 5u )
            return 0;
          if ( v30 == 16 )
          {
            v8 = *(_QWORD *)(v3 + 16);
            v3 = *(_QWORD *)v8;
          }
          LOBYTE(v10) = *(_BYTE *)(v3 + 8) == 11;
          LOBYTE(v8) = v9 == v12;
          return (unsigned int)v8 & v10;
        case ')':
        case '*':
          if ( v5 == 16 )
            v5 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
          v6 = 0;
          if ( v5 == 11 )
          {
            v31 = v9;
            LOBYTE(v13) = sub_15F4D40(v3);
            LOBYTE(v14) = v31 == v12;
            return v13 & v14;
          }
          return v6;
        case '+':
          if ( v5 == 16 )
            v5 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
          v28 = v8;
          v33 = v9;
          if ( (unsigned __int8)(v5 - 1) > 5u )
            return 0;
          LOBYTE(v21) = sub_15F4D40(v3);
          v6 = v21;
          if ( (_BYTE)v21 )
          {
            LOBYTE(v22) = v33 == v12;
            LOBYTE(v21) = v29 > v28;
            return v21 & v22;
          }
          return v6;
        case ',':
          if ( v5 == 16 )
            v5 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
          v27 = v8;
          v32 = v9;
          if ( (unsigned __int8)(v5 - 1) > 5u )
            return 0;
          LOBYTE(v19) = sub_15F4D40(v3);
          v6 = v19;
          if ( (_BYTE)v19 )
          {
            LOBYTE(v20) = v32 == v12;
            LOBYTE(v19) = v29 < v27;
            return v19 & v20;
          }
          return v6;
        case '-':
          v6 = 0;
          if ( (v30 == 16) != (v5 == 16) )
            return v6;
          if ( v5 != 16 )
            goto LABEL_78;
          if ( *(_QWORD *)(v3 + 32) == *(_QWORD *)(v4 + 32) )
          {
            v4 = **(_QWORD **)(v4 + 16);
LABEL_78:
            v6 = 0;
            if ( *(_BYTE *)(v4 + 8) == 15 )
            {
              if ( v30 == 16 )
                v3 = **(_QWORD **)(v3 + 16);
              LOBYTE(v6) = *(_BYTE *)(v3 + 8) == 11;
            }
          }
          break;
        case '.':
          v6 = 0;
          if ( (v30 == 16) != (v5 == 16) )
            return v6;
          if ( v5 != 16 )
            goto LABEL_70;
          if ( *(_QWORD *)(v3 + 32) == *(_QWORD *)(v4 + 32) )
          {
            v4 = **(_QWORD **)(v4 + 16);
LABEL_70:
            v6 = 0;
            if ( *(_BYTE *)(v4 + 8) == 11 )
            {
              if ( v30 == 16 )
                v3 = **(_QWORD **)(v3 + 16);
              LOBYTE(v6) = *(_BYTE *)(v3 + 8) == 15;
            }
          }
          break;
        case '/':
          v15 = v5;
          v16 = v4;
          if ( v5 == 16 )
          {
            v16 = **(_QWORD **)(v4 + 16);
            v15 = *(_BYTE *)(v16 + 8);
          }
          v17 = v30;
          if ( v15 != 15 )
            v16 = 0;
          v18 = v3;
          if ( v30 == 16 )
          {
            v18 = **(_QWORD **)(v3 + 16);
            v17 = *(_BYTE *)(v18 + 8);
          }
          if ( v17 != 15 )
            v18 = 0;
          if ( (v16 == 0) != (v18 == 0) )
            return 0;
          if ( v16 )
          {
            if ( *(_DWORD *)(v18 + 8) >> 8 != *(_DWORD *)(v16 + 8) >> 8 )
              return 0;
            goto LABEL_38;
          }
          v26 = sub_1643030(v4);
          LOBYTE(v6) = v26 == (unsigned int)sub_1643030(v3);
          return v6;
        case '0':
          v23 = v5;
          v24 = v4;
          if ( v5 == 16 )
          {
            v24 = **(_QWORD **)(v4 + 16);
            v23 = *(_BYTE *)(v24 + 8);
          }
          if ( v23 != 15 )
            return 0;
          v25 = v3;
          if ( v30 == 16 )
            v25 = **(_QWORD **)(v3 + 16);
          if ( *(_BYTE *)(v25 + 8) != 15 || *(_DWORD *)(v25 + 8) >> 8 == *(_DWORD *)(v24 + 8) >> 8 )
            return 0;
LABEL_38:
          if ( v5 == 16 )
          {
            v6 = 0;
            if ( v30 == 16 )
              LOBYTE(v6) = *(_QWORD *)(v3 + 32) == *(_QWORD *)(v4 + 32);
          }
          return v6;
        default:
          return 0;
      }
    }
  }
  return v6;
}

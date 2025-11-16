// Function: sub_38CDBE0
// Address: 0x38cdbe0
//
_BYTE *__fastcall sub_38CDBE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  _BYTE *v7; // rax
  _BYTE *result; // rax
  _DWORD *v9; // rdi
  __int64 v10; // rsi
  _BYTE *v11; // rax
  _BYTE *v12; // rax
  _BYTE *v13; // rax
  _BYTE *v14; // rax
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  _BYTE *v18; // rax
  _WORD *v19; // rdx
  _BYTE *v20; // rax
  _WORD *v21; // rdx
  _WORD *v22; // rdx
  _BYTE *v23; // rax
  _BYTE *v24; // rax
  _WORD *v25; // rdx
  _BYTE *v26; // rax
  _BYTE *v27; // rax
  __int64 v28; // rax
  _BYTE *v29; // rax
  _WORD *v30; // rdx
  _WORD *v31; // rdx
  _BYTE *v32; // rax
  _BYTE *v33; // rax
  _WORD *v34; // rdx
  _WORD *v35; // rdx
  _BYTE *v36; // rax

  while ( 2 )
  {
    v3 = a1;
    v4 = a3;
    v5 = a2;
    switch ( *(_DWORD *)a1 )
    {
      case 0:
        v9 = *(_DWORD **)(a1 + 24);
        if ( (unsigned int)(*v9 - 1) > 1 )
        {
          v12 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v12 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 40);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v12 + 1;
            *v12 = 40;
          }
          sub_38CDBE0(*(_QWORD *)(v3 + 24), a2, v4, 0);
          v13 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v13 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 41);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v13 + 1;
            *v13 = 41;
          }
        }
        else
        {
          sub_38CDBE0(v9, a2, a3, 0);
        }
        switch ( *(_DWORD *)(v3 + 16) )
        {
          case 0:
            v28 = *(_QWORD *)(v3 + 32);
            if ( *(_DWORD *)v28 == 1 )
            {
              v10 = *(_QWORD *)(v28 + 16);
              if ( v10 < 0 )
                goto LABEL_15;
            }
            v29 = *(_BYTE **)(v5 + 24);
            if ( (unsigned __int64)v29 >= *(_QWORD *)(v5 + 16) )
            {
              sub_16E7DE0(v5, 43);
            }
            else
            {
              *(_QWORD *)(v5 + 24) = v29 + 1;
              *v29 = 43;
            }
            break;
          case 1:
            v32 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v32 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 38);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v32 + 1;
              *v32 = 38;
            }
            break;
          case 2:
            v24 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v24 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 47);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v24 + 1;
              *v24 = 47;
            }
            break;
          case 3:
            v34 = *(_WORD **)(a2 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v34 <= 1u )
            {
              sub_16E7EE0(a2, "==", 2u);
            }
            else
            {
              *v34 = 15677;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
            break;
          case 4:
            v26 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v26 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 62);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v26 + 1;
              *v26 = 62;
            }
            break;
          case 5:
            v30 = *(_WORD **)(a2 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v30 <= 1u )
            {
              sub_16E7EE0(a2, ">=", 2u);
            }
            else
            {
              *v30 = 15678;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
            break;
          case 6:
            v22 = *(_WORD **)(a2 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v22 <= 1u )
            {
              sub_16E7EE0(a2, "&&", 2u);
            }
            else
            {
              *v22 = 9766;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
            break;
          case 7:
            v35 = *(_WORD **)(a2 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v35 <= 1u )
            {
              sub_16E7EE0(a2, "||", 2u);
            }
            else
            {
              *v35 = 31868;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
            break;
          case 8:
            v27 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v27 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 60);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v27 + 1;
              *v27 = 60;
            }
            break;
          case 9:
            v31 = *(_WORD **)(a2 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v31 <= 1u )
            {
              sub_16E7EE0(a2, "<=", 2u);
            }
            else
            {
              *v31 = 15676;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
            break;
          case 0xA:
            v23 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v23 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 37);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v23 + 1;
              *v23 = 37;
            }
            break;
          case 0xB:
            v33 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v33 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 42);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v33 + 1;
              *v33 = 42;
            }
            break;
          case 0xC:
            v25 = *(_WORD **)(a2 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v25 <= 1u )
            {
              sub_16E7EE0(a2, "!=", 2u);
            }
            else
            {
              *v25 = 15649;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
            break;
          case 0xD:
            v15 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v15 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 124);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v15 + 1;
              *v15 = 124;
            }
            break;
          case 0xE:
            v21 = *(_WORD **)(a2 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v21 <= 1u )
            {
              sub_16E7EE0(a2, "<<", 2u);
            }
            else
            {
              *v21 = 15420;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
            break;
          case 0xF:
          case 0x10:
            v19 = *(_WORD **)(a2 + 24);
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v19 <= 1u )
            {
              sub_16E7EE0(a2, ">>", 2u);
            }
            else
            {
              *v19 = 15934;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
            break;
          case 0x11:
            v20 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v20 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 45);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v20 + 1;
              *v20 = 45;
            }
            break;
          case 0x12:
            v36 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v36 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 94);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v36 + 1;
              *v36 = 94;
            }
            break;
          default:
            break;
        }
        a1 = *(_QWORD *)(v3 + 32);
        if ( (unsigned int)(*(_DWORD *)a1 - 1) <= 1 )
          goto LABEL_8;
        v16 = *(_BYTE **)(v5 + 24);
        if ( (unsigned __int64)v16 >= *(_QWORD *)(v5 + 16) )
        {
          sub_16E7DE0(v5, 40);
        }
        else
        {
          *(_QWORD *)(v5 + 24) = v16 + 1;
          *v16 = 40;
        }
        sub_38CDBE0(*(_QWORD *)(v3 + 32), v5, v4, 0);
        result = *(_BYTE **)(v5 + 24);
        if ( (unsigned __int64)result >= *(_QWORD *)(v5 + 16) )
        {
          result = (_BYTE *)sub_16E7DE0(v5, 41);
        }
        else
        {
          *(_QWORD *)(v5 + 24) = result + 1;
          *result = 41;
        }
        break;
      case 1:
        v10 = *(_QWORD *)(a1 + 16);
LABEL_15:
        result = (_BYTE *)sub_16E7AB0(v5, v10);
        break;
      case 2:
        result = (_BYTE *)sub_38E2490(*(_QWORD *)(a1 + 24), a2, a3);
        if ( *(_WORD *)(a1 + 16) )
          result = sub_38CDA50(a1, a2);
        break;
      case 3:
        v6 = *(_DWORD *)(a1 + 16);
        if ( v6 == 2 )
        {
          v18 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v18 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 126);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v18 + 1;
            *v18 = 126;
          }
        }
        else if ( v6 > 2 )
        {
          if ( v6 == 3 )
          {
            v11 = *(_BYTE **)(a2 + 24);
            if ( (unsigned __int64)v11 >= *(_QWORD *)(a2 + 16) )
            {
              sub_16E7DE0(a2, 43);
            }
            else
            {
              *(_QWORD *)(a2 + 24) = v11 + 1;
              *v11 = 43;
            }
          }
        }
        else if ( v6 )
        {
          v7 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v7 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 45);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v7 + 1;
            *v7 = 45;
          }
        }
        else
        {
          v17 = *(_BYTE **)(a2 + 24);
          if ( (unsigned __int64)v17 >= *(_QWORD *)(a2 + 16) )
          {
            sub_16E7DE0(a2, 33);
          }
          else
          {
            *(_QWORD *)(a2 + 24) = v17 + 1;
            *v17 = 33;
          }
        }
        a1 = *(_QWORD *)(a1 + 24);
        if ( *(_DWORD *)a1 )
        {
LABEL_8:
          a3 = v4;
          a2 = v5;
          continue;
        }
        v14 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v14 )
        {
          sub_16E7EE0(a2, "(", 1u);
        }
        else
        {
          *v14 = 40;
          ++*(_QWORD *)(a2 + 24);
        }
        sub_38CDBE0(*(_QWORD *)(v3 + 24), a2, v4, 0);
        result = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == result )
        {
          result = (_BYTE *)sub_16E7EE0(a2, ")", 1u);
        }
        else
        {
          *result = 41;
          ++*(_QWORD *)(a2 + 24);
        }
        break;
      case 4:
        result = (_BYTE *)(*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)(a1 - 8) + 24LL))(
                            a1 - 8,
                            a2,
                            a3);
        break;
    }
    return result;
  }
}

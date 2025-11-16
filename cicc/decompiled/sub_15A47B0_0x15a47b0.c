// Function: sub_15A47B0
// Address: 0x15a47b0
//
__int64 __fastcall sub_15A47B0(
        __int64 a1,
        _BYTE **a2,
        __int64 a3,
        __int64 **a4,
        char a5,
        __int64 a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v10; // r11
  __int64 **v13; // r12
  _BYTE **v14; // rcx
  _QWORD *v15; // rdx
  _QWORD *v16; // rax
  __int64 result; // rax
  _QWORD *v18; // r14
  __int64 ***v19; // r13
  unsigned __int16 v20; // ax
  __int64 ***v21; // r13
  __int64 **v22; // rdx
  unsigned __int64 v23; // rcx
  unsigned __int8 v24; // r8
  int v25; // eax
  int v26; // ebx
  unsigned int *v27; // rax
  __int64 v28; // rdx
  _DWORD *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  bool v33; // [rsp+Ch] [rbp-44h]
  bool v34; // [rsp+Ch] [rbp-44h]
  __int64 **v35; // [rsp+10h] [rbp-40h]
  int v36; // [rsp+28h] [rbp-28h] BYREF
  char v37; // [rsp+2Ch] [rbp-24h]

  v10 = a6;
  if ( a4 == *(__int64 ***)a1 )
  {
    v14 = &a2[a3];
    v15 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( a2 == v14 )
      return a1;
    v16 = a2;
    while ( *v15 == *v16 )
    {
      ++v16;
      v15 += 3;
      if ( v14 == v16 )
        return a1;
    }
  }
  v13 = 0;
  if ( a5 )
    v13 = a4;
  switch ( *(_WORD *)(a1 + 18) )
  {
    case ' ':
      v21 = (__int64 ***)*a2;
      v22 = (__int64 **)(a2 + 1);
      v23 = a3 - 1;
      v24 = (*(_BYTE *)(a1 + 17) & 2) != 0;
      v25 = *(_BYTE *)(a1 + 17) >> 1 >> 1;
      if ( v25 )
      {
        v26 = v25 - 1;
        if ( a6 )
        {
          v37 = 1;
        }
        else
        {
          v33 = (*(_BYTE *)(a1 + 17) & 2) != 0;
          v35 = v22;
          v31 = sub_16348C0(a1);
          v37 = 1;
          v24 = v33;
          v22 = v35;
          v23 = a3 - 1;
          v10 = v31;
        }
        v36 = v26;
      }
      else if ( a6 )
      {
        v37 = 0;
      }
      else
      {
        v34 = (*(_BYTE *)(a1 + 17) & 2) != 0;
        v32 = sub_16348C0(a1);
        v37 = 0;
        v23 = a3 - 1;
        v22 = (__int64 **)(a2 + 1);
        v24 = v34;
        v10 = v32;
      }
      result = sub_15A2E80(v10, (__int64)v21, v22, v23, v24, (__int64)&v36, (__int64)v13);
      break;
    case '$':
    case '%':
    case '&':
    case '\'':
    case '(':
    case ')':
    case '*':
    case '+':
    case ',':
    case '-':
    case '.':
    case '/':
    case '0':
      result = sub_15A46C0(*(unsigned __int16 *)(a1 + 18), (__int64 ***)*a2, a4, a5);
      break;
    case '3':
    case '4':
      v18 = a2[1];
      v19 = (__int64 ***)*a2;
      v20 = sub_1594720(a1);
      result = sub_15A37B0(v20, v19, v18, v13 != 0);
      break;
    case '7':
      result = sub_15A2DC0((__int64)*a2, (__int64 *)a2[1], (__int64)a2[2], (__int64)v13);
      break;
    case ';':
      result = sub_15A37D0(*a2, (__int64)a2[1], (__int64)v13);
      break;
    case '<':
      result = sub_15A3890((__int64 *)*a2, (__int64)a2[1], (__int64)a2[2], (__int64)v13);
      break;
    case '=':
      result = sub_15A3950((__int64)*a2, (__int64)a2[1], a2[2], v13);
      break;
    case '>':
      v27 = (unsigned int *)sub_1594710(a1);
      result = sub_15A3AE0(*a2, v27, v28, (__int64)v13);
      break;
    case '?':
      v29 = (_DWORD *)sub_1594710(a1);
      result = sub_15A3A20((__int64 *)*a2, (__int64 *)a2[1], v29, v30, (__int64)v13);
      break;
    default:
      result = sub_15A2A30(
                 (__int64 *)*(unsigned __int16 *)(a1 + 18),
                 (__int64 *)*a2,
                 (__int64)a2[1],
                 *(_BYTE *)(a1 + 17) >> 1,
                 (__int64)v13,
                 a7,
                 a8,
                 a9);
      break;
  }
  return result;
}

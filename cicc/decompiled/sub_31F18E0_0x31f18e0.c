// Function: sub_31F18E0
// Address: 0x31f18e0
//
unsigned __int64 __fastcall sub_31F18E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, size_t a5)
{
  __int64 v7; // rax
  void *v8; // rdi
  __int64 v9; // r14
  unsigned __int64 result; // rax
  __int64 v11; // rax
  void *v12; // rdi
  unsigned __int8 *v13; // rsi
  size_t v14; // r12
  size_t v15; // rdx
  char *v16; // r8
  __int64 v17; // rcx
  int v18; // r12d
  __int64 v19; // rsi
  __int64 v20; // [rsp-8h] [rbp-C8h]
  _QWORD v22[2]; // [rsp+10h] [rbp-B0h] BYREF
  char v23; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD *v24; // [rsp+30h] [rbp-90h] BYREF
  __int16 v25; // [rsp+50h] [rbp-70h]
  _QWORD v26[6]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD *v27; // [rsp+90h] [rbp-30h]

  if ( a5 == 7 )
  {
    if ( *(_DWORD *)a4 == 1986622064 && *(_WORD *)(a4 + 4) == 29793 && *(_BYTE *)(a4 + 6) == 101 )
    {
      result = *(unsigned int *)(sub_2E79000(*(__int64 **)(a1 + 232)) + 24);
      switch ( result )
      {
        case 0uLL:
          return result;
        case 1uLL:
        case 3uLL:
          v15 = 2;
          v16 = ".L";
          goto LABEL_23;
        case 2uLL:
        case 4uLL:
          v15 = 1;
          v16 = "L";
          goto LABEL_23;
        case 5uLL:
          v15 = 2;
          v16 = "L#";
          goto LABEL_23;
        case 6uLL:
          v15 = 1;
          v16 = "$";
          goto LABEL_23;
        case 7uLL:
          v15 = 3;
          v16 = "L..";
LABEL_23:
          v17 = *(_QWORD *)(a3 + 32);
          v13 = (unsigned __int8 *)v16;
          if ( *(_QWORD *)(a3 + 24) - v17 < v15 )
            return sub_CB6200(a3, v13, v15);
          LODWORD(result) = 0;
          do
          {
            v19 = (unsigned int)result;
            result = (unsigned int)(result + 1);
            *(_BYTE *)(v17 + v19) = v16[v19];
          }
          while ( (unsigned int)result < (unsigned int)v15 );
          *(_QWORD *)(a3 + 32) += v15;
          break;
        default:
          BUG();
      }
      return result;
    }
    if ( *(_DWORD *)a4 == 1835888483 && *(_WORD *)(a4 + 4) == 28261 && *(_BYTE *)(a4 + 6) == 116 )
    {
      v11 = *(_QWORD *)(a1 + 208);
      v12 = *(void **)(a3 + 32);
      v13 = *(unsigned __int8 **)(v11 + 48);
      v14 = *(_QWORD *)(v11 + 56);
      result = *(_QWORD *)(a3 + 24) - (_QWORD)v12;
      if ( v14 > result )
      {
        v15 = v14;
        return sub_CB6200(a3, v13, v15);
      }
      else if ( v14 )
      {
        result = (unsigned __int64)memcpy(v12, v13, v14);
        *(_QWORD *)(a3 + 32) += v14;
      }
      return result;
    }
LABEL_4:
    v22[0] = &v23;
    v27 = v22;
    v26[0] = &unk_49DD210;
    v22[1] = 0;
    v23 = 0;
    memset(&v26[1], 0, 32);
    v26[5] = 0x100000000LL;
    sub_CB5980((__int64)v26, 0, 0, 0);
    v7 = sub_904010((__int64)v26, "Unknown special formatter '");
    v8 = *(void **)(v7 + 32);
    v9 = v7;
    if ( *(_QWORD *)(v7 + 24) - (_QWORD)v8 < a5 )
    {
      v9 = sub_CB6200(v7, (unsigned __int8 *)a4, a5);
    }
    else if ( a5 )
    {
      memcpy(v8, (const void *)a4, a5);
      *(_QWORD *)(v9 + 32) += a5;
    }
    v20 = sub_904010(v9, "' for machine instr: ");
    sub_2E91850(a2, v20, 1u, 0, 0, 1, 0);
    v25 = 260;
    v24 = v27;
    sub_C64D30((__int64)&v24, 1u);
  }
  if ( a5 != 3 || *(_WORD *)a4 != 26997 || *(_BYTE *)(a4 + 2) != 100 )
    goto LABEL_4;
  if ( *(_QWORD *)(a1 + 960) != a2 || (v18 = *(_DWORD *)(a1 + 968), v18 != (unsigned int)sub_31DA6A0(a1)) )
  {
    ++*(_DWORD *)(a1 + 972);
    *(_QWORD *)(a1 + 960) = a2;
    *(_DWORD *)(a1 + 968) = sub_31DA6A0(a1);
  }
  return sub_CB59D0(a3, *(unsigned int *)(a1 + 972));
}

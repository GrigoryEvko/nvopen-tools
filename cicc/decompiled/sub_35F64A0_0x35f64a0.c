// Function: sub_35F64A0
// Address: 0x35f64a0
//
size_t __fastcall sub_35F64A0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  size_t result; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  char v11; // r13
  _QWORD *v12; // rdx
  __int64 v13; // rax
  bool v14; // cf
  size_t v15; // rdx
  char *v16; // rsi
  _QWORD *v17; // rdx
  __int64 v18; // rdx

  if ( !a5 )
    sub_C64ED0("Empty modifier in nvvm.red intrinsic", 1u);
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  result = strlen((const char *)a5);
  switch ( result )
  {
    case 3uLL:
      if ( *(_WORD *)a5 == 25971 && *(_BYTE *)(a5 + 2) == 109 )
      {
        v11 = v7 & 0xF;
        if ( v11 != 1 )
        {
          if ( v11 != 3 )
            sub_C64ED0("Invalid memory model ordering for nvvm.red", 1u);
          v12 = *(_QWORD **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v12 > 7u )
          {
            *v12 = 0x657361656C65722ELL;
            *(_QWORD *)(a4 + 32) += 8LL;
            return 0x657361656C65722ELL;
          }
          v15 = 8;
          v16 = ".release";
          return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
        }
        v17 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v17 <= 7u )
        {
          v15 = 8;
          v16 = ".relaxed";
          return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
        }
        *v17 = 0x646578616C65722ELL;
        *(_QWORD *)(a4 + 32) += 8LL;
        return 0x646578616C65722ELL;
      }
      break;
    case 5uLL:
      if ( *(_DWORD *)a5 == 1886348147 && *(_BYTE *)(a5 + 4) == 101 )
        return sub_35ED5A0(((unsigned int)v7 >> 4) & 7, a4);
      if ( *(_DWORD *)a5 == 1952870254 && *(_BYTE *)(a5 + 4) == 122 && (v7 & 0x4000) != 0 )
      {
        v10 = *(_QWORD *)(a4 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v10) > 5 )
        {
          *(_DWORD *)v10 = 1718578734;
          *(_WORD *)(v10 + 4) = 31348;
          *(_QWORD *)(a4 + 32) += 6LL;
          return 31348;
        }
        v15 = 6;
        v16 = ".noftz";
        return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
      }
      break;
    case 2uLL:
      if ( *(_WORD *)a5 == 28783 )
      {
        v9 = *(_QWORD *)(a4 + 32);
        switch ( ((unsigned int)v7 >> 10) & 0xF )
        {
          case 0u:
            result = *(_QWORD *)(a4 + 24) - v9;
            if ( result > 2 )
            {
              *(_BYTE *)(v9 + 2) = 100;
              *(_WORD *)v9 = 28257;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "and";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 1u:
            v13 = *(_QWORD *)(a4 + 24);
            v14 = v13 == v9;
            result = v13 - v9;
            if ( !v14 && result != 1 )
            {
              *(_WORD *)v9 = 29295;
              *(_QWORD *)(a4 + 32) += 2LL;
              return result;
            }
            v15 = 2;
            v16 = "or";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 2u:
            result = *(_QWORD *)(a4 + 24) - v9;
            if ( result > 2 )
            {
              *(_BYTE *)(v9 + 2) = 114;
              *(_WORD *)v9 = 28536;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "xor";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 3u:
            result = *(_QWORD *)(a4 + 24) - v9;
            if ( result > 2 )
            {
              *(_BYTE *)(v9 + 2) = 100;
              *(_WORD *)v9 = 25697;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "add";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 4u:
            result = *(_QWORD *)(a4 + 24) - v9;
            if ( result > 2 )
            {
              *(_BYTE *)(v9 + 2) = 99;
              *(_WORD *)v9 = 28265;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "inc";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 5u:
            result = *(_QWORD *)(a4 + 24) - v9;
            if ( result > 2 )
            {
              *(_BYTE *)(v9 + 2) = 99;
              *(_WORD *)v9 = 25956;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "dec";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 6u:
            result = *(_QWORD *)(a4 + 24) - v9;
            if ( result > 2 )
            {
              *(_BYTE *)(v9 + 2) = 110;
              *(_WORD *)v9 = 26989;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "min";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 7u:
            result = *(_QWORD *)(a4 + 24) - v9;
            if ( result > 2 )
            {
              *(_BYTE *)(v9 + 2) = 120;
              *(_WORD *)v9 = 24941;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "max";
            break;
          default:
            sub_C64ED0("Invalid reduction op for nvvm.red", 1u);
        }
        return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
      }
      break;
    default:
      if ( result == 4 && *(_DWORD *)a5 == 1701869940 )
      {
        v18 = *(_QWORD *)(a4 + 32);
        switch ( ((unsigned int)v7 >> 15) & 0xF )
        {
          case 0u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 2 )
            {
              *(_WORD *)v18 = 13154;
              *(_BYTE *)(v18 + 2) = 50;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "b32";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 1u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 2 )
            {
              *(_WORD *)v18 = 13922;
              *(_BYTE *)(v18 + 2) = 52;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "b64";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 2u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 2 )
            {
              *(_WORD *)v18 = 13173;
              *(_BYTE *)(v18 + 2) = 50;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "u32";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 3u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 2 )
            {
              *(_WORD *)v18 = 13941;
              *(_BYTE *)(v18 + 2) = 52;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "u64";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 4u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 2 )
            {
              *(_WORD *)v18 = 13171;
              *(_BYTE *)(v18 + 2) = 50;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "s32";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 5u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 2 )
            {
              *(_WORD *)v18 = 13939;
              *(_BYTE *)(v18 + 2) = 52;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "s64";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 6u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 2 )
            {
              *(_WORD *)v18 = 13158;
              *(_BYTE *)(v18 + 2) = 50;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "f32";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 7u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 2 )
            {
              *(_WORD *)v18 = 13926;
              *(_BYTE *)(v18 + 2) = 52;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "f64";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 8u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 2 )
            {
              *(_WORD *)v18 = 12646;
              *(_BYTE *)(v18 + 2) = 54;
              *(_QWORD *)(a4 + 32) += 3LL;
              return result;
            }
            v15 = 3;
            v16 = "f16";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 9u:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 4 )
            {
              *(_DWORD *)v18 = 2016817510;
              *(_BYTE *)(v18 + 4) = 50;
              *(_QWORD *)(a4 + 32) += 5LL;
              return result;
            }
            v15 = 5;
            v16 = "f16x2";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 0xAu:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 3 )
            {
              *(_DWORD *)v18 = 909207138;
              *(_QWORD *)(a4 + 32) += 4LL;
              return result;
            }
            v15 = 4;
            v16 = "bf16";
            return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
          case 0xBu:
            result = *(_QWORD *)(a4 + 24) - v18;
            if ( result > 5 )
            {
              *(_DWORD *)v18 = 909207138;
              *(_WORD *)(v18 + 4) = 12920;
              *(_QWORD *)(a4 + 32) += 6LL;
              return result;
            }
            v15 = 6;
            v16 = "bf16x2";
            break;
          default:
            sub_C64ED0("Invalid reduction type for nvvm.red", 1u);
        }
        return sub_CB6200(a4, (unsigned __int8 *)v16, v15);
      }
      break;
  }
  return result;
}

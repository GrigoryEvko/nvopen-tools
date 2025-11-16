// Function: sub_35EDAD0
// Address: 0x35edad0
//
unsigned __int64 __fastcall sub_35EDAD0(int a1, __int64 a2)
{
  _DWORD *v2; // rdx
  unsigned __int64 result; // rax
  _WORD *v4; // rdx
  _WORD *v5; // rdx
  char *v6; // rdx
  char *v7; // rax
  bool v8; // cf
  char *v9; // rdx
  char *v10; // rax
  char *v11; // rdx
  char *v12; // rax
  __int64 v13; // rdx
  _DWORD *v14; // rdx
  _DWORD *v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  _DWORD *v19; // rdx
  _DWORD *v20; // rdx
  _DWORD *v21; // rdx
  _DWORD *v22; // rdx

  switch ( a1 )
  {
    case 1:
      v4 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 1u )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"b1", 2u);
      }
      else
      {
        *v4 = 12642;
        *(_QWORD *)(a2 + 32) += 2LL;
        result = 12642;
      }
      break;
    case 2:
      v5 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v5 <= 1u )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"s4", 2u);
      }
      else
      {
        *v5 = 13427;
        *(_QWORD *)(a2 + 32) += 2LL;
        result = 13427;
      }
      break;
    case 3:
      v6 = *(char **)(a2 + 32);
      v7 = *(char **)(a2 + 24);
      v8 = v7 == v6;
      result = v7 - v6;
      if ( v8 || result == 1 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"u4", 2u);
      }
      else
      {
        *(_WORD *)v6 = 13429;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      break;
    case 4:
      v9 = *(char **)(a2 + 32);
      v10 = *(char **)(a2 + 24);
      v8 = v10 == v9;
      result = v10 - v9;
      if ( v8 || result == 1 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"s8", 2u);
      }
      else
      {
        *(_WORD *)v9 = 14451;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      break;
    case 5:
      v11 = *(char **)(a2 + 32);
      v12 = *(char **)(a2 + 24);
      v8 = v12 == v11;
      result = v12 - v11;
      if ( v8 || result == 1 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"u8", 2u);
      }
      else
      {
        *(_WORD *)v11 = 14453;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      break;
    case 6:
      v13 = *(_QWORD *)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - v13;
      if ( result <= 2 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"f16", 3u);
      }
      else
      {
        *(_BYTE *)(v13 + 2) = 54;
        *(_WORD *)v13 = 12646;
        *(_QWORD *)(a2 + 32) += 3LL;
      }
      break;
    case 7:
      v14 = *(_DWORD **)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - (_QWORD)v14;
      if ( result <= 3 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"bf16", 4u);
      }
      else
      {
        *v14 = 909207138;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      break;
    case 8:
      v15 = *(_DWORD **)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - (_QWORD)v15;
      if ( result <= 3 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"tf32", 4u);
      }
      else
      {
        *v15 = 842229364;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      break;
    case 9:
      v16 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v16) <= 2 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"f64", 3u);
      }
      else
      {
        *(_BYTE *)(v16 + 2) = 52;
        *(_WORD *)v16 = 13926;
        *(_QWORD *)(a2 + 32) += 3LL;
        result = 13926;
      }
      break;
    case 10:
      v17 = *(_QWORD *)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - v17;
      if ( result <= 2 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"f32", 3u);
      }
      else
      {
        *(_BYTE *)(v17 + 2) = 50;
        *(_WORD *)v17 = 13158;
        *(_QWORD *)(a2 + 32) += 3LL;
      }
      break;
    case 11:
      v18 = *(_QWORD *)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - v18;
      if ( result <= 2 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"s32", 3u);
      }
      else
      {
        *(_BYTE *)(v18 + 2) = 50;
        *(_WORD *)v18 = 13171;
        *(_QWORD *)(a2 + 32) += 3LL;
      }
      break;
    case 12:
      v19 = *(_DWORD **)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - (_QWORD)v19;
      if ( result <= 3 )
      {
        result = sub_CB6200(a2, "e5m2", 4u);
      }
      else
      {
        *v19 = 846017893;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      break;
    case 13:
      v20 = *(_DWORD **)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - (_QWORD)v20;
      if ( result <= 3 )
      {
        result = sub_CB6200(a2, (unsigned __int8 *)"e4m3", 4u);
      }
      else
      {
        *v20 = 862794853;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      break;
    case 15:
      v21 = *(_DWORD **)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - (_QWORD)v21;
      if ( result <= 3 )
      {
        result = sub_CB6200(a2, "e3m2", 4u);
      }
      else
      {
        *v21 = 846017381;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      break;
    case 16:
      v22 = *(_DWORD **)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - (_QWORD)v22;
      if ( result <= 3 )
      {
        result = sub_CB6200(a2, "e2m3", 4u);
      }
      else
      {
        *v22 = 862794341;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      break;
    case 17:
      v2 = *(_DWORD **)(a2 + 32);
      result = *(_QWORD *)(a2 + 24) - (_QWORD)v2;
      if ( result <= 3 )
      {
        result = sub_CB6200(a2, "e2m1", 4u);
      }
      else
      {
        *v2 = 829239909;
        *(_QWORD *)(a2 + 32) += 4LL;
      }
      break;
    default:
      sub_C64ED0("Wrong MMA element type", 1u);
  }
  return result;
}

// Function: sub_2E78A80
// Address: 0x2e78a80
//
__int64 __fastcall sub_2E78A80(_QWORD *a1, __int64 a2)
{
  char *v2; // r15
  int i; // r14d
  __int64 result; // rax
  size_t v7; // rax
  __int64 v8; // rcx
  size_t v9; // rdx
  __int64 v10; // r8
  char *v11; // r15
  size_t v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rsi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rax
  char *v17; // r15
  unsigned int v18; // ecx
  __int64 v19; // rsi
  __int64 v20; // [rsp+8h] [rbp-38h]

  v2 = (char *)byte_3F871B3;
  for ( i = 0; i != 12; ++i )
  {
    result = *a1 & (1LL << i);
    if ( !result )
      continue;
    v7 = strlen(v2);
    v8 = *(_QWORD *)(a2 + 32);
    v9 = v7;
    if ( v7 > *(_QWORD *)(a2 + 24) - v8 )
    {
      v10 = sub_CB6200(a2, (unsigned __int8 *)v2, v7);
    }
    else
    {
      v10 = a2;
      if ( v7 )
      {
        if ( (_DWORD)v7 )
        {
          v13 = 0;
          do
          {
            v14 = v13++;
            *(_BYTE *)(v8 + v14) = v2[v14];
          }
          while ( v13 < (unsigned int)v9 );
        }
        *(_QWORD *)(a2 + 32) += v9;
        v10 = a2;
      }
    }
    switch ( i )
    {
      case 1:
        v11 = "NoPHIs";
        break;
      case 2:
        v11 = "TracksLiveness";
        break;
      case 3:
        v11 = "NoVRegs";
        break;
      case 4:
        v11 = "FailedISel";
        break;
      case 5:
        v11 = "Legalized";
        break;
      case 6:
        v11 = "RegBankSelected";
        break;
      case 7:
        v11 = (char *)"Selected";
        break;
      case 8:
        v11 = "TiedOpsRewritten";
        break;
      case 9:
        v11 = "FailsVerification";
        break;
      case 10:
        v11 = "FailedRegAlloc";
        break;
      case 11:
        v11 = "TracksDebugUserValues";
        break;
      default:
        v11 = "IsSSA";
        break;
    }
    v20 = v10;
    v12 = strlen(v11);
    result = *(_QWORD *)(v20 + 32);
    if ( v12 <= *(_QWORD *)(v20 + 24) - result )
    {
      if ( (unsigned int)v12 >= 8 )
      {
        v15 = (result + 8) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)result = *(_QWORD *)v11;
        *(_QWORD *)(result + (unsigned int)v12 - 8) = *(_QWORD *)&v11[(unsigned int)v12 - 8];
        v16 = result - v15;
        v17 = &v11[-v16];
        result = ((_DWORD)v12 + (_DWORD)v16) & 0xFFFFFFF8;
        if ( (unsigned int)result >= 8 )
        {
          result = (unsigned int)result & 0xFFFFFFF8;
          v18 = 0;
          do
          {
            v19 = v18;
            v18 += 8;
            *(_QWORD *)(v15 + v19) = *(_QWORD *)&v17[v19];
          }
          while ( v18 < (unsigned int)result );
          *(_QWORD *)(v20 + 32) += v12;
          goto LABEL_9;
        }
      }
      else if ( (v12 & 4) != 0 )
      {
        *(_DWORD *)result = *(_DWORD *)v11;
        *(_DWORD *)(result + (unsigned int)v12 - 4) = *(_DWORD *)&v11[(unsigned int)v12 - 4];
      }
      else if ( (_DWORD)v12 )
      {
        *(_BYTE *)result = *v11;
        if ( (v12 & 2) != 0 )
          *(_WORD *)(result + (unsigned int)v12 - 2) = *(_WORD *)&v11[(unsigned int)v12 - 2];
      }
      *(_QWORD *)(v20 + 32) += v12;
    }
    else
    {
      result = sub_CB6200(v20, (unsigned __int8 *)v11, v12);
    }
LABEL_9:
    v2 = ", ";
  }
  return result;
}

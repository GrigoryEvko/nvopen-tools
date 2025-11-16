// Function: sub_380D6E0
// Address: 0x380d6e0
//
__int64 __fastcall sub_380D6E0(_QWORD *a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned int *v5; // rax
  __int64 v6; // rax
  int v7; // eax
  unsigned __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 v10; // r8
  unsigned int v12; // edx
  unsigned int v13; // edx
  unsigned int v14; // edx
  unsigned int v15; // edx
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned int v21; // edx

  v5 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v6 = *(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2];
  if ( (unsigned __int8)sub_3761870(a1, a2, *(_WORD *)v6, *(_QWORD *)(v6 + 8), 0) )
    return 0;
  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 > 278 )
  {
    switch ( v7 )
    {
      case 339:
        v8 = (unsigned __int64)sub_380D3C0((__int64)a1, a2, a4);
        v10 = v15;
        break;
      case 368:
        v8 = (unsigned __int64)sub_380C950((__int64)a1, a2, a3);
        v10 = v12;
        break;
      case 299:
        v8 = (unsigned __int64)sub_380D0E0((__int64)a1, a2, a4);
        v10 = v13;
        break;
      default:
        goto LABEL_24;
    }
  }
  else
  {
    if ( v7 > 206 )
    {
      switch ( v7 )
      {
        case 207:
          v8 = (unsigned __int64)sub_380CEA0((__int64)a1, a2);
          v10 = v19;
          goto LABEL_8;
        case 208:
          v8 = sub_380CFC0((__int64)a1, a2);
          v10 = v18;
          goto LABEL_8;
        case 226:
        case 227:
        case 275:
        case 276:
        case 277:
        case 278:
          v8 = (unsigned __int64)sub_380CB00((__int64)a1, a2, a4);
          v10 = v16;
          goto LABEL_8;
        case 228:
        case 229:
          v8 = (unsigned __int64)sub_380CBC0((__int64)a1, a2);
          v10 = v17;
          goto LABEL_8;
        case 233:
          v8 = (unsigned __int64)sub_380CCA0((__int64)a1, a2, a4);
          v10 = v21;
          goto LABEL_8;
        case 234:
          v8 = (unsigned __int64)sub_380C690((__int64)a1, a2, a4);
          v10 = v20;
          goto LABEL_8;
        default:
          goto LABEL_24;
      }
    }
    if ( v7 != 146 )
    {
      if ( v7 == 152 )
      {
        v8 = (unsigned __int64)sub_380CA20((__int64)a1, a2);
        v10 = v9;
        goto LABEL_8;
      }
LABEL_24:
      sub_C64ED0("Do not know how to promote this operator's operand!", 1u);
    }
    v8 = (unsigned __int64)sub_380CD80((__int64)a1, a2);
    v10 = v14;
  }
LABEL_8:
  if ( v8 )
    sub_3760E70((__int64)a1, a2, 0, v8, v10);
  return 0;
}

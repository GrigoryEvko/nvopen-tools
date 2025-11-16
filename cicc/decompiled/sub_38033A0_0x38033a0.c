// Function: sub_38033A0
// Address: 0x38033a0
//
__int64 __fastcall sub_38033A0(__int64 *a1, unsigned __int64 a2, unsigned int a3)
{
  unsigned int *v3; // rax
  __int64 v4; // rax
  unsigned int v5; // eax
  unsigned int v6; // r14d
  int v7; // eax
  __int64 v8; // rcx
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
  unsigned int v22; // edx
  unsigned int v23; // edx
  unsigned int v24; // edx

  v3 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v4 = *(_QWORD *)(*(_QWORD *)v3 + 48LL) + 16LL * v3[2];
  v5 = sub_3761870(a1, a2, *(_WORD *)v4, *(_QWORD *)(v4 + 8), 0);
  if ( (_BYTE)v5 )
    return 0;
  v6 = v5;
  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 > 234 )
  {
    switch ( v7 )
    {
      case 275:
        v8 = sub_3802D20((__int64)a1, a2);
        v10 = v17;
        goto LABEL_6;
      case 276:
        v8 = sub_3802EC0((__int64)a1, a2);
        v10 = v13;
        goto LABEL_6;
      case 277:
        v8 = sub_3803060((__int64)a1, a2);
        v10 = v12;
        goto LABEL_6;
      case 278:
        v8 = sub_3803200((__int64)a1, a2);
        v10 = v16;
        goto LABEL_6;
      case 299:
        v8 = sub_3802AD0(a1, a2);
        v10 = v15;
        goto LABEL_6;
      case 306:
        v8 = (__int64)sub_3801F80(a1, a2);
        v10 = v14;
        goto LABEL_6;
      default:
        goto LABEL_14;
    }
  }
  if ( v7 > 140 )
  {
    switch ( v7 )
    {
      case 141:
      case 142:
      case 226:
      case 227:
        v8 = sub_3802550(a1, a2);
        v10 = v18;
        goto LABEL_6;
      case 145:
      case 230:
        v8 = (__int64)sub_3802220((__int64)a1, a2);
        v10 = v20;
        goto LABEL_6;
      case 147:
      case 148:
      case 208:
        v8 = sub_3802960(a1, a2);
        v10 = v19;
        goto LABEL_6;
      case 152:
        v8 = (__int64)sub_3802130((__int64)a1, a2);
        v10 = v22;
        goto LABEL_6;
      case 156:
        v8 = sub_3847980(a1, a2);
        v10 = v21;
        goto LABEL_6;
      case 207:
        v8 = (__int64)sub_38027B0(a1, a2);
        v10 = v23;
        goto LABEL_6;
      case 234:
        v8 = sub_3847520(a1, a2);
        v10 = v24;
        goto LABEL_6;
      default:
        goto LABEL_14;
    }
  }
  if ( v7 != 53 )
LABEL_14:
    sub_C64ED0("Do not know how to expand this operator's operand!", 1u);
  v8 = sub_38480C0(a1, a2);
  v10 = v9;
LABEL_6:
  if ( !v8 )
  {
    return 0;
  }
  else if ( a2 == v8 )
  {
    return 1;
  }
  else
  {
    sub_3760E70((__int64)a1, a2, 0, v8, v10);
  }
  return v6;
}

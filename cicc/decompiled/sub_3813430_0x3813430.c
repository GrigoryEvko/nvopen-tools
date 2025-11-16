// Function: sub_3813430
// Address: 0x3813430
//
__int64 __fastcall sub_3813430(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned int *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // eax
  unsigned __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 v13; // r8
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

  v5 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v6 = *(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2];
  if ( !(unsigned __int8)sub_3761870(a1, a2, *(_WORD *)v6, *(_QWORD *)(v6 + 8), 0) )
  {
    v10 = *(_DWORD *)(a2 + 24);
    if ( v10 <= 234 )
    {
      if ( v10 > 140 )
      {
        switch ( v10 )
        {
          case 141:
          case 142:
          case 226:
          case 227:
            v11 = (unsigned __int64)sub_3812230(a1, a2, a4);
            v13 = v17;
            goto LABEL_10;
          case 146:
          case 233:
            v11 = (unsigned __int64)sub_3811EB0((__int64)a1, a2, a4);
            v13 = v19;
            goto LABEL_10;
          case 152:
            v11 = (unsigned __int64)sub_3811C40(a1, a2, a4, a3, v7, v8, v9);
            v13 = v20;
            goto LABEL_10;
          case 207:
            v11 = (unsigned __int64)sub_38128A0(a1, a2);
            v13 = v23;
            goto LABEL_10;
          case 208:
            v11 = sub_3812C60(a1, a2);
            v13 = v22;
            goto LABEL_10;
          case 228:
          case 229:
            v11 = (unsigned __int64)sub_3812690(a1, a2, a4);
            v13 = v18;
            goto LABEL_10;
          case 234:
            v11 = (unsigned __int64)sub_3811AA0((__int64)a1, a2, a4);
            v13 = v21;
            goto LABEL_10;
          default:
            break;
        }
      }
LABEL_26:
      sub_C64ED0("Do not know how to soft promote this operator's operand!", 1u);
    }
    if ( v10 == 368 )
    {
      v11 = (unsigned __int64)sub_3811B70((__int64)a1, a2, a3);
      v13 = v24;
    }
    else if ( v10 <= 368 )
    {
      if ( v10 == 299 )
      {
        v11 = (unsigned __int64)sub_3813040((__int64)a1, a2);
        v13 = v16;
      }
      else
      {
        if ( v10 != 339 )
          goto LABEL_26;
        v11 = (unsigned __int64)sub_3813100((__int64)a1, a2);
        v13 = v15;
      }
    }
    else
    {
      if ( v10 != 393 && v10 != 394 )
        goto LABEL_26;
      v11 = sub_3813200((__int64)a1, a2, a3, v7, v8, v9);
      v13 = v12;
    }
LABEL_10:
    if ( v11 )
      sub_3760E70((__int64)a1, a2, 0, v11, v13);
  }
  return 0;
}

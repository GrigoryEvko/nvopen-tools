// Function: sub_382B700
// Address: 0x382b700
//
__int64 __fastcall sub_382B700(__int64 *a1, unsigned __int64 a2, unsigned __int32 a3, __m128i a4)
{
  unsigned int *v5; // rax
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned int v10; // r15d
  int v11; // eax
  __int64 v12; // rbx
  __int64 v13; // rcx
  unsigned int v14; // edx
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned __int64 *v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // edx
  unsigned int v24; // edx
  unsigned int v25; // edx
  unsigned int v26; // edx
  unsigned int v27; // edx
  unsigned int v28; // edx
  unsigned int v29; // edx
  unsigned int v30; // edx
  unsigned int v31; // edx
  unsigned int v32; // edx
  unsigned int v33; // edx
  unsigned int v34; // edx
  unsigned int v35; // edx
  __int64 v36; // [rsp+130h] [rbp-50h] BYREF
  __int64 v37; // [rsp+138h] [rbp-48h]
  __int64 v38; // [rsp+140h] [rbp-40h] BYREF
  int v39; // [rsp+148h] [rbp-38h]

  v5 = (unsigned int *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v6 = *(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2];
  v7 = sub_3761870(a1, a2, *(_WORD *)v6, *(_QWORD *)(v6 + 8), 0);
  if ( (_BYTE)v7 )
    return 0;
  v10 = v7;
  v11 = *(_DWORD *)(a2 + 24);
  v12 = 0;
  if ( v11 <= 234 )
  {
    if ( v11 > 142 )
    {
      switch ( v11 )
      {
        case 143:
        case 144:
        case 220:
        case 221:
          v13 = sub_3829070(a1, a2);
          v12 = v25;
          goto LABEL_8;
        case 156:
          v13 = sub_3847980(a1, a2);
          v12 = v33;
          goto LABEL_8;
        case 157:
          v13 = sub_3848350(a1, a2);
          v12 = v34;
          goto LABEL_8;
        case 167:
          v13 = sub_38488B0(a1, a2);
          v12 = v30;
          goto LABEL_8;
        case 168:
          goto LABEL_20;
        case 184:
        case 185:
          v13 = sub_34680A0((_DWORD *)*a1, a2, (_QWORD *)a1[1], a4);
          v12 = v26;
          goto LABEL_8;
        case 190:
        case 191:
        case 192:
        case 193:
        case 194:
          v13 = (__int64)sub_3828FF0((__int64)a1, a2);
          v12 = v24;
          goto LABEL_8;
        case 207:
          v13 = (__int64)sub_3828AA0(a1, a2);
          v12 = v29;
          goto LABEL_8;
        case 208:
          v13 = (__int64)sub_3828C30(a1, a2);
          v12 = v27;
          goto LABEL_8;
        case 209:
          v13 = (__int64)sub_3828D10((__int64)a1, a2);
          v12 = v28;
          goto LABEL_8;
        case 216:
          v13 = (__int64)sub_3829FC0((__int64)a1, a2, a4);
          v12 = v31;
          goto LABEL_8;
        case 234:
          v13 = sub_3847520(a1, a2);
          v12 = v32;
          goto LABEL_8;
        default:
          goto LABEL_13;
      }
    }
    if ( v11 > 23 )
    {
      if ( v11 == 53 )
      {
        v13 = sub_38480C0(a1, a2);
        v12 = v14;
        goto LABEL_8;
      }
LABEL_13:
      sub_C64ED0("Do not know how to expand this operator's operand!", 1u);
    }
    if ( v11 <= 21 )
      goto LABEL_13;
    v21 = *(unsigned __int64 **)(a2 + 40);
    LODWORD(v37) = 0;
    v39 = 0;
    v22 = v21[1];
    v36 = 0;
    v38 = 0;
    sub_375E510((__int64)a1, *v21, v22, (__int64)&v36, (__int64)&v38);
    v13 = (__int64)sub_33EBEE0((_QWORD *)a1[1], (__int64 *)a2, v36, v37);
    goto LABEL_8;
  }
  if ( v11 == 368 )
  {
    v13 = sub_38481D0(a1, a2);
    v12 = v35;
    goto LABEL_8;
  }
  if ( v11 <= 368 )
  {
    switch ( v11 )
    {
      case 306:
        v13 = (__int64)sub_3828910(a1, a2);
        v12 = v23;
        break;
      case 339:
        v13 = (__int64)sub_382A090((__int64)a1, a2);
        v12 = v17;
        break;
      case 299:
        v13 = sub_3829320(a1, a2);
        v12 = v20;
        break;
      default:
        goto LABEL_13;
    }
    goto LABEL_8;
  }
  if ( v11 == 394 )
  {
LABEL_25:
    v13 = sub_382B260((__int64)a1, a2, a3, a4);
    v12 = v18;
    goto LABEL_8;
  }
  if ( v11 <= 394 )
  {
    if ( v11 != 393 )
      goto LABEL_13;
    goto LABEL_25;
  }
  if ( v11 != 469 )
  {
    if ( v11 == 492 )
    {
LABEL_20:
      v13 = (__int64)sub_3828F10((__int64)a1, a2);
      v12 = v16;
      goto LABEL_8;
    }
    if ( v11 != 466 )
      goto LABEL_13;
  }
  v13 = (__int64)sub_382A130((__int64)a1, a2, a3, v8, v9);
  v12 = v19;
LABEL_8:
  if ( !v13 )
    return 0;
  if ( a2 == v13 )
    return 1;
  else
    sub_3760E70((__int64)a1, a2, 0, v13, v12);
  return v10;
}

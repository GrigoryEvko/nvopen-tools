// Function: sub_8CAE10
// Address: 0x8cae10
//
__int64 __fastcall sub_8CAE10(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r14
  int v4; // r13d
  __int64 i; // rbx
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rsi
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 v24; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // rax
  __int64 *v27; // r13
  __int64 *j; // rbx
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // r12
  __int64 v34; // rsi
  _QWORD *v35; // rbx
  _QWORD *v36; // r12
  _QWORD *v37; // rax
  __int64 v38; // r13
  int v39; // eax
  __int64 v40; // r8
  int v41; // eax
  __int64 v42; // r8
  _BOOL4 v43; // eax
  _QWORD *v44; // rbx
  _QWORD *v45; // r8
  __int64 v46; // r14
  __int64 **v47; // rdi
  __int64 **v48; // rsi
  __int64 *v49; // r12
  __int64 *v50; // r13
  int v51; // eax
  _QWORD *v52; // [rsp-60h] [rbp-60h]
  __int64 *v53; // [rsp-60h] [rbp-60h]
  _QWORD *v54; // [rsp-58h] [rbp-58h]
  __int64 *v55; // [rsp-58h] [rbp-58h]
  __int64 v56; // [rsp-50h] [rbp-50h]
  __int64 v57; // [rsp-50h] [rbp-50h]
  __int64 v58; // [rsp-50h] [rbp-50h]
  __int64 v59; // [rsp-50h] [rbp-50h]
  __int64 v60; // [rsp-50h] [rbp-50h]
  char v61; // [rsp-39h] [rbp-39h] BYREF

  result = *(_QWORD *)(a1 + 32);
  if ( !result )
    return result;
  v3 = *(_QWORD *)result;
  if ( a1 == *(_QWORD *)result )
    return result;
  result = (unsigned int)*(unsigned __int8 *)(v3 + 140) - 9;
  if ( (unsigned __int8)(*(_BYTE *)(v3 + 140) - 9) > 2u )
    return result;
  result = sub_8D23B0(a1);
  if ( (_DWORD)result )
    return result;
  result = sub_8D2490(a1);
  if ( !(_DWORD)result )
    return result;
  v4 = sub_8D23B0(v3);
  if ( v4 || !(unsigned int)sub_8D2490(v3) )
  {
    result = sub_8D2490(a1);
    if ( (_DWORD)result )
      return (__int64)sub_8C9FB0(a1, 1u);
  }
  else
  {
    for ( i = *(_QWORD *)(a1 + 160); i; i = *(_QWORD *)(i + 112) )
    {
      if ( (*(_BYTE *)(i + 144) & 0x50) != 0x40 )
        break;
    }
    v6 = *(_QWORD *)(v3 + 160);
    if ( v6 )
    {
      while ( (*(_BYTE *)(v6 + 144) & 0x50) == 0x40 )
      {
        v6 = *(_QWORD *)(v6 + 112);
        if ( !v6 )
          goto LABEL_25;
      }
LABEL_17:
      if ( i )
      {
        sub_8CBB20(8, i, v6);
        do
          i = *(_QWORD *)(i + 112);
        while ( i && (*(_BYTE *)(i + 144) & 0x50) == 0x40 );
        while ( 1 )
        {
          v6 = *(_QWORD *)(v6 + 112);
          if ( !v6 )
            break;
          if ( (*(_BYTE *)(v6 + 144) & 0x50) != 0x40 )
            goto LABEL_17;
        }
      }
    }
LABEL_25:
    result = (__int64)&dword_4F077C4;
    if ( dword_4F077C4 == 2 )
    {
      v54 = *(_QWORD **)(*(_QWORD *)(a1 + 168) + 152LL);
      v7 = v54[34];
      v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v3 + 168) + 152LL) + 272LL);
      v52 = *(_QWORD **)(*(_QWORD *)(v3 + 168) + 152LL);
      if ( !v8 || !v7 )
      {
LABEL_43:
        v9 = sub_8C6310(v54[13]);
        v10 = sub_8C6310(v52[13]);
        v11 = (__int64 *)v10;
        if ( v9 && v10 )
        {
          do
          {
            sub_8CA500(v9, (__int64)v11);
            if ( (unsigned __int8)(*(_BYTE *)(v9 + 140) - 9) <= 2u && *(char *)(v9 + 177) < 0 )
            {
              v13 = *(_QWORD *)v9;
              v14 = *v11;
              v15 = *(_BYTE *)(*v11 + 80);
              switch ( *(_BYTE *)(*(_QWORD *)v9 + 80LL) )
              {
                case 4:
                case 5:
                  v16 = *(_QWORD *)(*(_QWORD *)(v13 + 96) + 80LL);
                  goto LABEL_53;
                case 6:
                  v16 = *(_QWORD *)(*(_QWORD *)(v13 + 96) + 32LL);
                  goto LABEL_53;
                case 9:
                case 0xA:
                  v16 = *(_QWORD *)(*(_QWORD *)(v13 + 96) + 56LL);
                  goto LABEL_53;
                case 0x13:
                case 0x14:
                case 0x15:
                case 0x16:
                  v16 = *(_QWORD *)(v13 + 88);
LABEL_53:
                  switch ( v15 )
                  {
                    case 4:
                    case 5:
                      goto LABEL_63;
                    case 6:
                      goto LABEL_65;
                    case 9:
                    case 10:
                      goto LABEL_62;
                    case 19:
                    case 20:
                    case 21:
                    case 22:
                      goto LABEL_54;
                    default:
                      goto LABEL_46;
                  }
                default:
                  v16 = 0;
                  switch ( v15 )
                  {
                    case 4:
                    case 5:
LABEL_63:
                      v17 = *(_QWORD *)(*(_QWORD *)(v14 + 96) + 80LL);
                      break;
                    case 6:
LABEL_65:
                      v17 = *(_QWORD *)(*(_QWORD *)(v14 + 96) + 32LL);
                      break;
                    case 9:
                    case 10:
LABEL_62:
                      v17 = *(_QWORD *)(*(_QWORD *)(v14 + 96) + 56LL);
                      break;
                    case 19:
                    case 20:
                    case 21:
                    case 22:
LABEL_54:
                      v17 = *(_QWORD *)(v14 + 88);
                      break;
                    default:
                      v17 = 0;
                      break;
                  }
                  if ( v16 )
                  {
                    if ( v17 )
                    {
                      v18 = *(_QWORD *)(v16 + 104);
                      if ( v18 )
                      {
                        v19 = *(_QWORD *)(v17 + 104);
                        if ( v19 )
                          sub_8CBB20(59, v18, v19);
                      }
                    }
                  }
                  break;
              }
            }
LABEL_46:
            v9 = sub_8C6310(*(_QWORD *)(v9 + 112));
            v12 = sub_8C6310(v11[14]);
            v11 = (__int64 *)v12;
          }
          while ( v9 && v12 );
        }
        v20 = sub_8C6270(v54[18]);
        v21 = sub_8C6270(v52[18]);
        if ( v21 && v20 )
        {
          do
          {
            if ( ((*(_BYTE *)(v21 + 193) ^ *(_BYTE *)(v20 + 193)) & 0x10) != 0
              || ((*(_BYTE *)(v21 + 195) ^ *(_BYTE *)(v20 + 195)) & 1) != 0 )
            {
              v23 = *(__int64 **)(a1 + 32);
              v24 = a1;
              if ( v23 )
                v24 = *v23;
              sub_8C6700((__int64 *)a1, (unsigned int *)(v24 + 64), 0x42Au, 0x425u);
              v25 = *(_QWORD **)(v20 + 32);
              if ( !v25 || *v25 != v20 )
                sub_8C7090(11, v20);
              v26 = *(_QWORD **)(v21 + 32);
              if ( !v26 || *v26 != v21 )
                sub_8C7090(11, v21);
            }
            else
            {
              sub_8CC0D0(v20, v21);
            }
            if ( (*(_BYTE *)(v20 + 195) & 0x28) == 8 && (*(_BYTE *)(v21 + 195) & 0x28) == 8 )
              sub_8CBF50(*(_QWORD *)v20, *(_QWORD *)v21);
            v20 = sub_8C6270(*(_QWORD *)(v20 + 112));
            v22 = sub_8C6270(*(_QWORD *)(v21 + 112));
            v21 = v22;
          }
          while ( v20 && v22 );
        }
        v27 = (__int64 *)v54[14];
        for ( j = (__int64 *)v52[14]; v27; j = (__int64 *)j[14] )
        {
          if ( !j )
            break;
          v30 = *v27;
          v31 = *j;
          if ( *v27 && v31 && (*(_QWORD *)(v30 + 96) == 0) != (*(_QWORD *)(v31 + 96) == 0) )
          {
            v56 = *j;
            sub_8C6700(v27, (unsigned int *)(v31 + 48), 0x42Au, 0x425u);
            v29 = v56;
            if ( !v27[4] )
            {
              sub_8C7090(7, (__int64)v27);
              v29 = v56;
            }
          }
          else
          {
            v57 = *j;
            sub_8CBB20(7, v27, j);
            v29 = v57;
          }
          if ( *(char *)(a1 + 177) < 0 )
          {
            v58 = v29;
            sub_8CBF50(v30, v29);
            if ( (*(_BYTE *)(v30 + 81) & 2) != 0 && (*(_BYTE *)(v58 + 81) & 2) != 0 )
              sub_8CA8D0(*(_QWORD **)(*(_QWORD *)(v30 + 88) + 208LL), *(_QWORD **)(*(_QWORD *)(v58 + 88) + 208LL));
          }
          v27 = (__int64 *)v27[14];
        }
        v32 = v54[12];
        v33 = v52[12];
        if ( v33 && v32 )
        {
          do
          {
            sub_8CBB20(2, v32, v33);
            v32 = *(_QWORD *)(v32 + 120);
            v33 = *(_QWORD *)(v33 + 120);
          }
          while ( v32 && v33 );
        }
        v34 = *(_QWORD *)(a1 + 168);
        v35 = sub_8C62C0(*(_QWORD **)(v34 + 136));
        v36 = sub_8C62C0(*(_QWORD **)(*(_QWORD *)(v3 + 168) + 136LL));
        if ( v36 && v35 )
        {
          do
          {
            v38 = v36[1];
            v59 = v35[1];
            v55 = *(__int64 **)v59;
            v53 = *(__int64 **)v38;
            if ( sub_8C7520((__int64 **)v59, (__int64 **)v38) && (!*(_QWORD *)(v59 + 32) || !*(_QWORD *)(v38 + 32)) )
            {
              if ( (v39 = sub_8C6B40((__int64)v55), v40 = v59, v39) && (v51 = sub_8C6B40((__int64)v53), v40 = v59, v51)
                || *(char *)(v40 + 192) < 0 && *(char *)(v38 + 192) < 0 )
              {
                v60 = v40;
                if ( (unsigned int)sub_8C7F70((__int64)v55, (__int64)v53) )
                {
                  v41 = sub_8DE890(*(_QWORD *)(v60 + 152), *(_QWORD *)(v38 + 152), 260, 0);
                  v42 = v60;
                  if ( v41 || sub_72F8B0((__int64 **)v60) && (v43 = sub_72F8B0((__int64 **)v38), v42 = v60, v43) )
                    sub_8CC0D0(v42, v38);
                }
              }
            }
            v35 = sub_8C62C0((_QWORD *)*v35);
            v37 = sub_8C62C0((_QWORD *)*v36);
            v36 = v37;
          }
          while ( v35 && v37 );
          v34 = *(_QWORD *)(a1 + 168);
        }
        v44 = sub_8C63A0(*(_QWORD **)(v34 + 144));
        result = (__int64)sub_8C63A0(v45);
        v46 = result;
        if ( result && v44 )
        {
          do
          {
            v47 = (__int64 **)v44[1];
            v48 = *(__int64 ***)(v46 + 8);
            v49 = *v47;
            v50 = *v48;
            if ( sub_8C7520(v47, v48) && (!*(_QWORD *)(v44[1] + 32LL) || !*(_QWORD *)(*(_QWORD *)(v46 + 8) + 32LL)) )
            {
              if ( (unsigned int)sub_8C7F70((__int64)v49, (__int64)v50) )
                sub_8CA500(v44[1], *(_QWORD *)(v46 + 8));
            }
            v44 = sub_8C63A0((_QWORD *)*v44);
            result = (__int64)sub_8C63A0(*(_QWORD **)v46);
            v46 = result;
          }
          while ( v44 && result );
        }
        return result;
      }
      while ( 1 )
      {
        if ( *(_BYTE *)(v7 + 120) == *(_BYTE *)(v8 + 120) )
        {
          if ( !v4 )
          {
            v61 = 59;
            if ( dword_4F07590 && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v7 + 80LL) - 19) > 1u )
              sub_87D510(*(_QWORD *)v7, &v61);
            sub_8CBB20(59, v7, v8);
            if ( v61 == 59 )
              sub_8CAA20((__int64 *)v7, (_QWORD *)v8);
            goto LABEL_32;
          }
        }
        else if ( !v4 )
        {
          sub_8C6700((__int64 *)v7, (unsigned int *)(v8 + 64), 0x42Au, 0x489u);
        }
        v4 = 1;
        sub_8C7090(59, v7);
        sub_8C99B0(*(_QWORD *)v7, 1);
LABEL_32:
        v7 = *(_QWORD *)(v7 + 112);
        v8 = *(_QWORD *)(v8 + 112);
        if ( !v7 || !v8 )
          goto LABEL_43;
      }
    }
  }
  return result;
}

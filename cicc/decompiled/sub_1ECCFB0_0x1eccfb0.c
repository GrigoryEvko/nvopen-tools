// Function: sub_1ECCFB0
// Address: 0x1eccfb0
//
unsigned __int64 __fastcall sub_1ECCFB0(_QWORD *a1)
{
  __int64 v2; // r14
  __int64 v3; // rax
  unsigned int v4; // ebx
  int *v5; // r10
  unsigned int v6; // r15d
  _DWORD *v7; // rsi
  unsigned __int64 result; // rax
  __int64 v9; // rcx
  int v10; // r10d
  __int64 v11; // rdx
  _DWORD *v12; // rdi
  _QWORD *v13; // rdx
  _QWORD *v14; // rcx
  unsigned int v15; // esi
  _QWORD *v16; // rax
  _BOOL4 v17; // r10d
  __int64 v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // rdx
  unsigned int v21; // ecx
  _QWORD *v22; // rax
  _BOOL4 v23; // r10d
  __int64 v24; // rax
  _QWORD *v25; // rdx
  _QWORD *v26; // rcx
  unsigned int v27; // esi
  _QWORD *v28; // rax
  _BOOL4 v29; // r10d
  __int64 v30; // rax
  _QWORD *v31; // [rsp+8h] [rbp-68h]
  _BOOL4 v32; // [rsp+8h] [rbp-68h]
  _QWORD *v33; // [rsp+8h] [rbp-68h]
  _QWORD *v34; // [rsp+8h] [rbp-68h]
  _QWORD *v35; // [rsp+8h] [rbp-68h]
  _QWORD *v36; // [rsp+10h] [rbp-60h]
  _QWORD *v37; // [rsp+10h] [rbp-60h]
  _QWORD *v38; // [rsp+10h] [rbp-60h]
  _QWORD *v39; // [rsp+10h] [rbp-60h]
  _QWORD *v40; // [rsp+10h] [rbp-60h]
  _QWORD *v41; // [rsp+10h] [rbp-60h]
  _QWORD *v42; // [rsp+18h] [rbp-58h]
  _BOOL4 v43; // [rsp+20h] [rbp-50h]
  _BOOL4 v44; // [rsp+20h] [rbp-50h]
  int v45; // [rsp+24h] [rbp-4Ch]
  __int64 v46; // [rsp+28h] [rbp-48h]
  unsigned int v47; // [rsp+34h] [rbp-3Ch] BYREF
  __int64 v48[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = *a1;
  v3 = *(_QWORD *)(*a1 + 168LL) - *(_QWORD *)(*a1 + 160LL);
  v48[0] = *a1;
  v47 = 0;
  v4 = -1171354717 * (v3 >> 3);
  if ( v4 )
  {
    v5 = (int *)&v47;
    while ( 1 )
    {
      v7 = *(_DWORD **)(v2 + 192);
      if ( v7 == sub_1ECAFE0(*(_DWORD **)(v2 + 184), (__int64)v7, v5) )
        break;
      v6 = v47 + 1;
      v47 = v6;
      if ( v4 <= v6 )
        goto LABEL_6;
    }
    v6 = v47;
  }
  else
  {
    v6 = 0;
  }
LABEL_6:
  result = sub_1ECCC00((__int64)v48);
  v45 = result;
  v42 = a1 + 14;
  if ( (_DWORD)result != v6 )
  {
    while ( 1 )
    {
      v46 = 88LL * v6;
      v9 = *(_QWORD *)(*a1 + 160LL) + v46;
      v10 = *(_DWORD *)(v9 + 16);
      if ( *(_QWORD *)(v9 + 72) - *(_QWORD *)(v9 + 64) <= 8u )
      {
        v47 = v6;
        switch ( v10 )
        {
          case 2:
            sub_1ECB700(a1 + 7, &v47);
            break;
          case 3:
            sub_1ECB700(a1 + 1, &v47);
            break;
          case 1:
            sub_1ECB700(a1 + 13, &v47);
            v25 = (_QWORD *)a1[3];
            v26 = a1 + 2;
            if ( !v25 )
            {
LABEL_85:
              v25 = v26;
              if ( v26 == (_QWORD *)a1[4] )
              {
                v29 = 1;
LABEL_58:
                v44 = v29;
                v33 = v25;
                v38 = v26;
                v30 = sub_22077B0(40);
                *(_DWORD *)(v30 + 32) = v6;
                sub_220F040(v44, v30, v33, v38);
                ++a1[6];
LABEL_59:
                result = *(_QWORD *)(*a1 + 160LL);
                *(_DWORD *)(result + v46 + 16) = 3;
                goto LABEL_24;
              }
LABEL_71:
              v35 = v26;
              v41 = v25;
              if ( v6 <= *(_DWORD *)(sub_220EF80(v25) + 32) )
                goto LABEL_59;
              v25 = v41;
              v26 = v35;
              if ( !v41 )
                goto LABEL_59;
              v29 = 1;
              if ( v35 == v41 )
                goto LABEL_58;
              goto LABEL_74;
            }
            while ( 1 )
            {
LABEL_52:
              v27 = *((_DWORD *)v25 + 8);
              v28 = (_QWORD *)v25[3];
              if ( v27 > v6 )
                v28 = (_QWORD *)v25[2];
              if ( !v28 )
                break;
              v25 = v28;
            }
            if ( v6 < v27 )
            {
              if ( (_QWORD *)a1[4] != v25 )
                goto LABEL_71;
            }
            else if ( v27 >= v6 )
            {
              goto LABEL_59;
            }
            v29 = 1;
            if ( v26 == v25 )
              goto LABEL_58;
LABEL_74:
            v29 = *((_DWORD *)v25 + 8) > v6;
            goto LABEL_58;
          default:
            break;
        }
        v25 = (_QWORD *)a1[3];
        v26 = a1 + 2;
        if ( !v25 )
          goto LABEL_85;
        goto LABEL_52;
      }
      v11 = *(unsigned int *)(v9 + 20);
      if ( *(_DWORD *)(v9 + 24) >= (unsigned int)v11 )
      {
        v47 = 0;
        v12 = *(_DWORD **)(v9 + 32);
        if ( &v12[v11] == sub_1ECB090(v12, (__int64)&v12[v11], (int *)&v47) )
        {
          v47 = v6;
          if ( v10 == 2 )
          {
            sub_1ECB700(a1 + 7, &v47);
            v20 = (_QWORD *)a1[15];
            if ( !v20 )
              goto LABEL_81;
          }
          else
          {
            if ( v10 == 3 )
            {
              sub_1ECB700(a1 + 1, &v47);
            }
            else if ( v10 == 1 )
            {
              sub_1ECB700(a1 + 13, &v47);
            }
            v20 = (_QWORD *)a1[15];
            if ( !v20 )
            {
LABEL_81:
              v20 = v42;
              if ( (_QWORD *)a1[16] == v42 )
              {
                v20 = v42;
                v23 = 1;
LABEL_44:
                v32 = v23;
                v37 = v20;
                v24 = sub_22077B0(40);
                *(_DWORD *)(v24 + 32) = v6;
                sub_220F040(v32, v24, v37, v42);
                ++a1[18];
LABEL_45:
                result = *(_QWORD *)(*a1 + 160LL);
                *(_DWORD *)(result + v46 + 16) = 1;
                goto LABEL_24;
              }
LABEL_61:
              v39 = v20;
              if ( *(_DWORD *)(sub_220EF80(v20) + 32) >= v6 )
                goto LABEL_45;
              v20 = v39;
              if ( !v39 )
                goto LABEL_45;
              v23 = 1;
              if ( v39 == v42 )
                goto LABEL_44;
              goto LABEL_64;
            }
          }
          while ( 1 )
          {
            v21 = *((_DWORD *)v20 + 8);
            v22 = (_QWORD *)v20[3];
            if ( v21 > v6 )
              v22 = (_QWORD *)v20[2];
            if ( !v22 )
              break;
            v20 = v22;
          }
          if ( v6 < v21 )
          {
            if ( (_QWORD *)a1[16] != v20 )
              goto LABEL_61;
          }
          else if ( v21 >= v6 )
          {
            goto LABEL_45;
          }
          v23 = 1;
          if ( v20 == v42 )
            goto LABEL_44;
LABEL_64:
          v23 = *((_DWORD *)v20 + 8) > v6;
          goto LABEL_44;
        }
      }
      v47 = v6;
      if ( v10 == 2 )
        break;
      if ( v10 == 3 )
      {
        sub_1ECB700(a1 + 1, &v47);
        goto LABEL_13;
      }
      if ( v10 != 1 )
        goto LABEL_13;
      sub_1ECB700(a1 + 13, &v47);
      v13 = (_QWORD *)a1[9];
      v14 = a1 + 8;
      if ( !v13 )
      {
LABEL_76:
        v13 = v14;
        if ( v14 != (_QWORD *)a1[10] )
          goto LABEL_66;
        v17 = 1;
        goto LABEL_22;
      }
      while ( 1 )
      {
LABEL_16:
        v15 = *((_DWORD *)v13 + 8);
        v16 = (_QWORD *)v13[3];
        if ( v15 > v6 )
          v16 = (_QWORD *)v13[2];
        if ( !v16 )
          break;
        v13 = v16;
      }
      if ( v6 < v15 )
      {
        if ( (_QWORD *)a1[10] == v13 )
        {
LABEL_21:
          v17 = 1;
          if ( v13 != v14 )
            goto LABEL_69;
        }
        else
        {
LABEL_66:
          v34 = v14;
          v40 = v13;
          if ( v6 <= *(_DWORD *)(sub_220EF80(v13) + 32) )
            goto LABEL_23;
          v13 = v40;
          v14 = v34;
          if ( !v40 )
            goto LABEL_23;
          v17 = 1;
          if ( v40 != v34 )
LABEL_69:
            v17 = *((_DWORD *)v13 + 8) > v6;
        }
LABEL_22:
        v43 = v17;
        v31 = v14;
        v36 = v13;
        v18 = sub_22077B0(40);
        *(_DWORD *)(v18 + 32) = v6;
        sub_220F040(v43, v18, v36, v31);
        ++a1[12];
        goto LABEL_23;
      }
      if ( v15 < v6 )
        goto LABEL_21;
LABEL_23:
      result = *(_QWORD *)(*a1 + 160LL);
      *(_DWORD *)(result + v46 + 16) = 2;
LABEL_24:
      v47 = ++v6;
      if ( v4 > v6 )
      {
        while ( 1 )
        {
          v19 = *(_QWORD *)(v2 + 192);
          result = (unsigned __int64)sub_1ECAFE0(*(_DWORD **)(v2 + 184), v19, (int *)&v47);
          if ( v19 == result )
            break;
          result = v47;
          v6 = v47 + 1;
          v47 = v6;
          if ( v4 <= v6 )
            goto LABEL_29;
        }
        v6 = v47;
      }
LABEL_29:
      if ( v6 == v45 )
        return result;
    }
    sub_1ECB700(a1 + 7, &v47);
LABEL_13:
    v13 = (_QWORD *)a1[9];
    v14 = a1 + 8;
    if ( !v13 )
      goto LABEL_76;
    goto LABEL_16;
  }
  return result;
}

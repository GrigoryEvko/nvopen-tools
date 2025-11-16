// Function: sub_5EAAC0
// Address: 0x5eaac0
//
__int64 __fastcall sub_5EAAC0(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 i; // rax
  __int64 result; // rax
  __int64 *v5; // rbx
  char v6; // dl
  _QWORD *v7; // rdx
  __int64 v8; // rsi
  _QWORD *v9; // r15
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 k; // rdi
  __int64 v14; // rcx
  __int64 v15; // r13
  int v16; // r12d
  _BOOL4 v17; // r14d
  __int64 v18; // rax
  _QWORD *v19; // rbx
  _BOOL4 v20; // r15d
  _QWORD *v21; // r14
  char *v22; // rax
  char v23; // dl
  __int64 v24; // r12
  __int64 j; // rax
  _QWORD *v26; // rax
  _QWORD *v27; // rsi
  unsigned __int64 v28; // rcx
  _QWORD *v29; // rax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rax
  bool v34; // r12
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 m; // rax
  __int64 v38; // rcx
  _QWORD *v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rax
  unsigned int v42; // [rsp+0h] [rbp-50h]
  __int64 *v43; // [rsp+0h] [rbp-50h]
  _BOOL4 v45; // [rsp+Ch] [rbp-44h]
  unsigned int v47; // [rsp+14h] [rbp-3Ch]
  __int64 v48; // [rsp+18h] [rbp-38h]
  int v49; // [rsp+18h] [rbp-38h]

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = *(_QWORD *)(*(_QWORD *)i + 96LL);
  v5 = *(__int64 **)(result + 56);
  if ( v5 )
  {
    v6 = *(_BYTE *)(result + 180);
    if ( a3 )
    {
      if ( (v6 & 0x20) != 0 )
        return result;
      *(_BYTE *)(result + 180) = v6 | 0x20;
      if ( (*(_BYTE *)(a1 + 177) & 0x30) != 0x30 )
      {
LABEL_7:
        v47 = 0;
        v45 = a2 != 0;
LABEL_8:
        v48 = 0;
        while ( 1 )
        {
          v15 = v5[2];
          k = v5[1];
          v16 = sub_5E9520(k, v15);
          result = v47;
          if ( v47 )
          {
            v7 = 0;
            if ( (*(_BYTE *)(k + 178) & 5) == 0 )
            {
              v7 = (_QWORD *)v47;
              result = v45;
            }
          }
          else
          {
            result = v45;
            v7 = 0;
          }
          v8 = *((unsigned __int8 *)v5 + 184);
          v9 = (_QWORD *)v5[16];
          v10 = v8;
          if ( (v8 & 2) == 0 )
            break;
          if ( (*(_BYTE *)(v15 + 80) == 20 || (_DWORD)result != 1) && a3 )
          {
            result = (__int64)&dword_4D047B0;
            v34 = v16 == 0;
            if ( dword_4D047B0 || (k = v15, result = sub_893570(v15, v8, v7, v14, v8), (_DWORD)result) )
            {
              if ( v9 )
              {
                k = v15;
                result = sub_893600(v15, v9, v5[17], 1, v10);
              }
              v8 = *((unsigned __int8 *)v5 + 184);
              if ( (v8 & 8) != 0 && v34 )
              {
LABEL_77:
                k = **(_QWORD **)(*(_QWORD *)(v15 + 88) + 176LL);
                result = sub_894C00(k, v8, v7, v14, v10);
                v8 = *((unsigned __int8 *)v5 + 184);
              }
            }
            else
            {
              v8 = *((unsigned __int8 *)v5 + 184);
              if ( (v8 & 8) != 0 && v34 && v45 )
                goto LABEL_77;
            }
          }
LABEL_14:
          if ( (v8 & 0x10) != 0 )
          {
            v11 = v5[1];
            if ( v11 == v48 )
            {
LABEL_23:
              k = *(_QWORD *)(v15 + 88);
              v8 = (v8 & 0x20) != 0;
              result = sub_5EAA30(k, v8);
              goto LABEL_24;
            }
            if ( v11 )
            {
              if ( !v48 )
              {
LABEL_22:
                sub_866000(v11, a2, 1);
                LOBYTE(v8) = *((_BYTE *)v5 + 184);
                v48 = v5[1];
                goto LABEL_23;
              }
              if ( dword_4F07588 )
              {
                v12 = *(_QWORD *)(v48 + 32);
                if ( *(_QWORD *)(v11 + 32) == v12 )
                {
                  if ( v12 )
                    goto LABEL_23;
                }
              }
            }
            else if ( !v48 )
            {
              goto LABEL_22;
            }
            sub_866010(v11, v8, v7, v48, v10);
            v11 = v5[1];
            goto LABEL_22;
          }
LABEL_24:
          v5 = (__int64 *)*v5;
          if ( !v5 )
          {
            if ( v48 )
              return sub_866010(k, v8, v7, v14, v10);
            return result;
          }
        }
        if ( !v9 && (v8 & 8) == 0 )
          goto LABEL_14;
        v14 = *(unsigned __int8 *)(v15 + 80);
        v17 = (_BYTE)v14 == 17 || (unsigned __int8)(v14 - 10) <= 1u;
        if ( !(_DWORD)v7 )
        {
          v14 = a3;
          if ( a3 )
            goto LABEL_24;
          v14 = v48;
          if ( k != v48 )
          {
            if ( k )
            {
              if ( !v48 )
                goto LABEL_41;
              v7 = (_QWORD *)dword_4F07588;
              if ( !dword_4F07588 || (v7 = *(_QWORD **)(v48 + 32), *(_QWORD **)(k + 32) != v7) || !v7 )
              {
LABEL_40:
                v49 = result;
                sub_866010(k, v8, v7, v14, v8);
                k = v5[1];
                LODWORD(result) = v49;
                goto LABEL_41;
              }
            }
            else
            {
              if ( v48 )
                goto LABEL_40;
LABEL_41:
              v42 = result;
              sub_866000(k, a2, 1);
              v48 = v5[1];
              result = v42;
            }
          }
          if ( (_DWORD)result && *(_BYTE *)(v15 + 80) == 10 && !v16 )
          {
            for ( ; v9; v9 = (_QWORD *)*v9 )
            {
              k = (__int64)(v9 + 1);
              result = sub_7AEA70(v9 + 1);
            }
            if ( !*(_QWORD *)(v15 + 96) )
              goto LABEL_92;
          }
          else
          {
            v18 = sub_877FE0(v5[2]);
            sub_8600D0(1, *((unsigned int *)v5 + 16), v18, 0);
            k = v5[3];
            if ( k )
              sub_886000(k);
            if ( v9 )
            {
              v43 = v5;
              v19 = v9;
              v20 = v17;
              while ( 1 )
              {
                k = (__int64)(v19 + 1);
                v21 = (_QWORD *)v19[6];
                sub_7BC000(v19 + 1);
                if ( !v16 )
                  goto LABEL_51;
                for ( j = *(_QWORD *)(*(_QWORD *)(v15 + 88) + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                  ;
                v26 = *(_QWORD **)(j + 168);
                v27 = (_QWORD *)*v26;
                if ( !v21 )
                  break;
                v28 = 0;
                do
                {
                  v21 = (_QWORD *)*v21;
                  ++v28;
                }
                while ( v21 );
                if ( v27 )
                  goto LABEL_65;
                v30 = 0;
LABEL_67:
                if ( v28 > v30 )
                  goto LABEL_52;
                v31 = v30 - v28;
                v21 = v27;
                v32 = v31 - 1;
                if ( v31 )
                {
                  do
                    v21 = (_QWORD *)*v21;
                  while ( v32-- != 0 );
                }
LABEL_51:
                if ( v21 )
                {
                  k = (__int64)v21;
                  sub_6794F0(v21, v15, v16 == 0);
                  v19 = (_QWORD *)*v19;
                  if ( !v19 )
                    goto LABEL_53;
                }
                else
                {
LABEL_52:
                  v19 = (_QWORD *)*v19;
                  v20 = 0;
                  if ( !v19 )
                  {
LABEL_53:
                    v5 = v43;
                    v17 = v20;
                    goto LABEL_54;
                  }
                }
              }
              if ( !v27 )
                goto LABEL_52;
              v28 = 0;
LABEL_65:
              v29 = (_QWORD *)*v26;
              v30 = 0;
              do
              {
                v29 = (_QWORD *)*v29;
                ++v30;
              }
              while ( v29 );
              goto LABEL_67;
            }
LABEL_54:
            if ( (v5[23] & 8) != 0 )
            {
              k = *(_QWORD *)(v15 + 88);
              v22 = *(char **)(*(_QWORD *)(*(_QWORD *)(k + 152) + 168LL) + 56LL);
              v23 = *v22;
              if ( (*v22 & 0x20) != 0 )
              {
                v24 = *((_QWORD *)v22 + 1);
                *((_QWORD *)v22 + 1) = 0;
                *v22 = v23 & 0xDF;
                if ( v24 )
                {
                  sub_625150(k, v24, 0);
                  k = v24;
                  sub_7AEB40(v24);
                }
              }
            }
            result = sub_863FC0();
          }
LABEL_90:
          if ( v17 )
          {
            v35 = v5[13];
            if ( v35 )
            {
              for ( k = *(_QWORD *)(*(_QWORD *)(v15 + 88) + 152LL); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
                ;
              result = sub_73BC00(k, v35);
            }
          }
LABEL_92:
          v8 = *((unsigned __int8 *)v5 + 184);
          goto LABEL_14;
        }
        if ( v16 || (_BYTE)v14 != 10 )
        {
          v36 = v8 & 8;
          if ( (_DWORD)v36 )
          {
            k = v15;
            result = sub_894C00(v15, v36, v7, v14, v10);
          }
          while ( v9 )
          {
            k = (__int64)(v9 + 1);
            result = sub_7AEA70(v9 + 1);
            v9 = (_QWORD *)*v9;
          }
          goto LABEL_90;
        }
        if ( !a3 && v9 )
        {
          k = v9[6];
          sub_648BD0(k, v9[2] + 8LL, v7, v14, v8);
          for ( m = v5[1]; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
            ;
          v38 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)m + 96LL) + 80LL);
          v39 = v9;
          do
          {
            v39[5] = *(_QWORD *)(v38 + 32);
            v39 = (_QWORD *)*v39;
          }
          while ( v39 );
          v40 = v9;
          do
          {
            v7 = v40;
            v40 = (_QWORD *)*v40;
          }
          while ( v40 );
          switch ( *(_BYTE *)(v15 + 80) )
          {
            case 4:
            case 5:
              v41 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 80LL);
              break;
            case 6:
              v41 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 32LL);
              break;
            case 9:
            case 0xA:
              v41 = *(_QWORD *)(*(_QWORD *)(v15 + 96) + 56LL);
              break;
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
              v41 = *(_QWORD *)(v15 + 88);
              break;
            default:
              BUG();
          }
          *v7 = *(_QWORD *)(v41 + 288);
          *(_QWORD *)(v41 + 288) = v9;
          result = dword_4D047B0;
          v14 = dword_4D047B0 == 0;
          v17 = dword_4D047B0 == 0;
        }
        else
        {
          k = dword_4D047B0;
          if ( dword_4D047B0 )
          {
            v8 = v5[16];
            k = v15;
            result = sub_893600(v15, v8, v5[3], 0, v10);
            LOBYTE(v10) = *((_BYTE *)v5 + 184);
          }
          v10 &= 8u;
          if ( (_DWORD)v10 )
          {
            k = v15;
            result = sub_894C00(v15, v8, v7, v14, v10);
          }
          if ( a3 )
          {
LABEL_89:
            v5[16] = 0;
            goto LABEL_90;
          }
          result = dword_4D047B0;
        }
        if ( (_DWORD)result )
          goto LABEL_90;
        goto LABEL_89;
      }
    }
    else
    {
      if ( (v6 & 0x10) != 0 )
        return result;
      *(_BYTE *)(result + 180) = v6 | 0x10;
      if ( (*(_BYTE *)(a1 + 177) & 0x30) != 0x30 )
        goto LABEL_7;
    }
    v47 = 1;
    v45 = 0;
    goto LABEL_8;
  }
  return result;
}

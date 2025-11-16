// Function: sub_2153AE0
// Address: 0x2153ae0
//
__int64 __fastcall sub_2153AE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rsi
  _BYTE *k; // rdx
  int *v8; // rax
  unsigned int v9; // r13d
  int v10; // ebx
  unsigned int v11; // r12d
  unsigned int v12; // r15d
  __int64 v13; // rdi
  __int64 *v14; // rax
  __int64 v15; // r13
  __int64 v16; // rdx
  __int64 v17; // r9
  __int64 v18; // rax
  int v19; // ecx
  __int64 v20; // r12
  __int64 m; // rbx
  unsigned int v22; // ebx
  __int64 v23; // rcx
  int v24; // r15d
  unsigned int v25; // eax
  __int64 v26; // rdi
  __int64 v27; // rsi
  __int64 v28; // r12
  __int64 v29; // r13
  __int64 *v30; // r12
  int i; // ebx
  __int64 v32; // rdi
  _BYTE *v33; // rax
  __int64 v34; // rdi
  _WORD *v35; // rdx
  __int64 v36; // rdi
  _BYTE *v37; // rax
  unsigned int v38; // eax
  __int64 *j; // rbx
  int v40; // ebx
  __int64 v41; // rax
  __int64 *v42; // r12
  __int64 v43; // r13
  __int64 v44; // rdi
  _BYTE *v45; // rax
  __int64 v46; // rdi
  _WORD *v47; // rdx
  int v48; // ecx
  __int64 v49; // rdi
  _BYTE *v50; // rax
  bool v51; // [rsp+17h] [rbp-149h]
  unsigned int v52; // [rsp+20h] [rbp-140h]
  __int64 v53; // [rsp+28h] [rbp-138h]
  __int64 v54; // [rsp+28h] [rbp-138h]
  unsigned int v55; // [rsp+28h] [rbp-138h]
  __int64 v56[4]; // [rsp+30h] [rbp-130h] BYREF
  __int64 v57[4]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v58[4]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v59[4]; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v60[4]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v61[4]; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v62[4]; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v63[10]; // [rsp+110h] [rbp-50h] BYREF

  if ( *(_DWORD *)a1 )
  {
    v8 = *(int **)(a1 + 32);
    if ( *(_BYTE *)(a1 + 185) )
    {
      v22 = *v8;
      sub_214B770(v56, "0XFF");
      sub_214B770(v57, "0xFF00");
      sub_214B770(v58, "0xFF0000");
      sub_214B770(v59, "0xFF000000");
      sub_214B770(v60, "0xFF00000000");
      sub_214B770(v61, "0xFF0000000000");
      sub_214B770(v62, "0xFF000000000000");
      sub_214B770(v63, "0xFF00000000000000");
      v24 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 176) + 232LL) + 936LL) == 0 ? 4 : 8;
      if ( *(_DWORD *)(a1 + 4) )
      {
        v52 = 0;
        v55 = 0;
LABEL_37:
        v25 = v55;
        if ( v55 == v22 )
        {
LABEL_38:
          v26 = *(_QWORD *)(a1 + 176);
          v55 = v24 + v25;
          v27 = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * v52);
          v28 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 8LL * v52);
          if ( *(_BYTE *)(v27 + 16) > 3u )
          {
            v40 = 0;
            v41 = sub_2153350(v26, v28, 0, v23);
            v42 = v56;
            v43 = v41;
            do
            {
              v49 = sub_16E7EE0(*(_QWORD *)(a1 + 168), (char *)*v42, v42[1]);
              v50 = *(_BYTE **)(v49 + 24);
              if ( *(_BYTE **)(v49 + 16) == v50 )
              {
                sub_16E7EE0(v49, "(", 1u);
              }
              else
              {
                *v50 = 40;
                ++*(_QWORD *)(v49 + 24);
              }
              sub_21537F0(*(_QWORD *)(a1 + 176), v43, *(_QWORD *)(a1 + 168), v48);
              v44 = *(_QWORD *)(a1 + 168);
              v45 = *(_BYTE **)(v44 + 24);
              if ( *(_BYTE **)(v44 + 16) == v45 )
              {
                sub_16E7EE0(v44, ")", 1u);
              }
              else
              {
                *v45 = 41;
                ++*(_QWORD *)(v44 + 24);
              }
              if ( v24 - 1 != v40 )
              {
                v46 = *(_QWORD *)(a1 + 168);
                v47 = *(_WORD **)(v46 + 24);
                if ( *(_QWORD *)(v46 + 16) - (_QWORD)v47 <= 1u )
                {
                  sub_16E7EE0(v46, ", ", 2u);
                }
                else
                {
                  *v47 = 8236;
                  *(_QWORD *)(v46 + 24) += 2LL;
                }
              }
              ++v40;
              v42 += 4;
            }
            while ( v24 != v40 );
          }
          else
          {
            v51 = 0;
            v29 = sub_396EAF0(v26, v27);
            if ( *(_BYTE *)(*(_QWORD *)v28 + 8LL) == 15 )
              v51 = *(_DWORD *)(*(_QWORD *)v28 + 8LL) >> 8 != 0;
            v30 = v56;
            for ( i = 0; i != v24; ++i )
            {
              v36 = sub_16E7EE0(*(_QWORD *)(a1 + 168), (char *)*v30, v30[1]);
              v37 = *(_BYTE **)(v36 + 24);
              if ( *(_BYTE **)(v36 + 16) == v37 )
              {
                sub_16E7EE0(v36, "(", 1u);
              }
              else
              {
                *v37 = 40;
                ++*(_QWORD *)(v36 + 24);
              }
              if ( *(_BYTE *)(a1 + 184) && *(_BYTE *)(v27 + 16) && !v51 )
              {
                sub_1263B40(*(_QWORD *)(a1 + 168), "generic(");
                sub_38E2490(v29, *(_QWORD *)(a1 + 168), *(_QWORD *)(*(_QWORD *)(a1 + 176) + 240LL));
                sub_1263B40(*(_QWORD *)(a1 + 168), ")");
              }
              else
              {
                sub_38E2490(v29, *(_QWORD *)(a1 + 168), *(_QWORD *)(*(_QWORD *)(a1 + 176) + 240LL));
              }
              v32 = *(_QWORD *)(a1 + 168);
              v33 = *(_BYTE **)(v32 + 24);
              if ( *(_BYTE **)(v32 + 16) == v33 )
              {
                sub_16E7EE0(v32, ")", 1u);
              }
              else
              {
                *v33 = 41;
                ++*(_QWORD *)(v32 + 24);
              }
              if ( v24 - 1 != i )
              {
                v34 = *(_QWORD *)(a1 + 168);
                v35 = *(_WORD **)(v34 + 24);
                if ( *(_QWORD *)(v34 + 16) - (_QWORD)v35 <= 1u )
                {
                  sub_16E7EE0(v34, ", ", 2u);
                }
                else
                {
                  *v35 = 8236;
                  *(_QWORD *)(v34 + 24) += 2LL;
                }
              }
              v30 += 4;
            }
          }
          v38 = *(_DWORD *)(a1 + 4);
          ++v52;
          v22 = v38 + 1;
          if ( *(_DWORD *)a1 > v52 )
            v22 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4LL * v52);
          v23 = v55;
          if ( v55 < v38 )
            goto LABEL_59;
        }
        else
        {
          while ( 1 )
          {
            sub_16E7A90(*(_QWORD *)(a1 + 168), *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + v55++));
            v23 = v55;
            if ( v55 >= *(_DWORD *)(a1 + 4) )
              break;
LABEL_59:
            if ( !(_DWORD)v23 )
              goto LABEL_37;
            sub_1263B40(*(_QWORD *)(a1 + 168), ", ");
            v25 = v55;
            if ( v55 == v22 )
              goto LABEL_38;
          }
        }
      }
      for ( j = v63; ; j -= 4 )
      {
        if ( (__int64 *)*j != j + 2 )
          j_j___libc_free_0(*j, j[2] + 1);
        result = (__int64)(j - 4);
        if ( j == v56 )
          break;
      }
    }
    else
    {
      v9 = *v8;
      result = *(unsigned int *)(a1 + 4);
      v10 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 176) + 232LL) + 936LL) == 0 ? 4 : 8;
      if ( (_DWORD)result )
      {
        v11 = 0;
        v12 = 0;
LABEL_11:
        if ( v9 != v12 )
        {
LABEL_12:
          v13 = *(_QWORD *)(a1 + 168);
          v14 = (__int64 *)(*(_QWORD *)(a1 + 8) + v12);
          if ( v10 == 4 )
            sub_16E7A90(v13, *(unsigned int *)v14);
          else
            sub_16E7AD0(v13, *v14);
          result = *(unsigned int *)(a1 + 4);
          goto LABEL_15;
        }
        while ( 1 )
        {
          v15 = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * v11);
          v16 = *(_QWORD *)(a1 + 112);
          if ( *(_BYTE *)(v15 + 16) > 3u )
          {
            v18 = sub_2153350(*(_QWORD *)(a1 + 176), *(_QWORD *)(v16 + 8LL * v11), 0, a4);
            sub_21537F0(*(_QWORD *)(a1 + 176), v18, *(_QWORD *)(a1 + 168), v19);
          }
          else if ( (v53 = *(_QWORD *)(v16 + 8LL * v11),
                     v17 = sub_396EAF0(*(_QWORD *)(a1 + 176), *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * v11)),
                     *(_BYTE *)(*(_QWORD *)v53 + 8LL) == 15)
                 && *(_DWORD *)(*(_QWORD *)v53 + 8LL) >> 8
                 || !*(_BYTE *)(a1 + 184)
                 || !*(_BYTE *)(v15 + 16) )
          {
            sub_38E2490(v17, *(_QWORD *)(a1 + 168), *(_QWORD *)(*(_QWORD *)(a1 + 176) + 240LL));
          }
          else
          {
            v54 = v17;
            sub_1263B40(*(_QWORD *)(a1 + 168), "generic(");
            sub_38E2490(v54, *(_QWORD *)(a1 + 168), *(_QWORD *)(*(_QWORD *)(a1 + 176) + 240LL));
            sub_1263B40(*(_QWORD *)(a1 + 168), ")");
          }
          result = *(unsigned int *)(a1 + 4);
          ++v11;
          v9 = result + 1;
          if ( *(_DWORD *)a1 > v11 )
          {
            a4 = *(_QWORD *)(a1 + 32);
            v9 = *(_DWORD *)(a4 + 4LL * v11);
          }
LABEL_15:
          v12 += v10;
          if ( v12 >= (unsigned int)result )
            break;
          if ( !v12 )
            goto LABEL_11;
          sub_1263B40(*(_QWORD *)(a1 + 168), ", ");
          if ( v9 != v12 )
            goto LABEL_12;
        }
      }
    }
  }
  else
  {
    result = *(unsigned int *)(a1 + 4);
    if ( (_DWORD)result )
    {
      v6 = *(_QWORD *)(a1 + 8);
      result = (unsigned int)(result - 1);
      for ( k = (_BYTE *)(v6 + result); !*k; --k )
      {
        if ( !(_DWORD)result )
          return result;
        result = (unsigned int)(result - 1);
      }
      v20 = (unsigned int)result;
      for ( m = 0; ; ++m )
      {
        result = sub_16E7A90(*(_QWORD *)(a1 + 168), *(unsigned __int8 *)(v6 + m));
        if ( m == v20 )
          break;
        if ( (_DWORD)m != -1 )
          sub_1263B40(*(_QWORD *)(a1 + 168), ", ");
        v6 = *(_QWORD *)(a1 + 8);
      }
    }
  }
  return result;
}

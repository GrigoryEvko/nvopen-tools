// Function: sub_2FD5E60
// Address: 0x2fd5e60
//
__int64 *__fastcall sub_2FD5E60(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  __int64 *result; // rax
  __int64 v7; // r12
  __int64 v8; // rdx
  int v9; // esi
  unsigned int i; // ebx
  unsigned int v11; // r13d
  int v12; // r14d
  __int64 v13; // rdx
  __int64 v14; // rsi
  unsigned int v15; // edi
  int *v16; // rax
  int v17; // r9d
  __int64 v18; // r14
  __int64 v19; // rbx
  __int64 v20; // rsi
  int v21; // eax
  __int64 v22; // rbx
  __int64 *v23; // rdx
  __int64 *j; // r13
  int v25; // ecx
  __int64 *v27; // [rsp+10h] [rbp-B0h]
  __int64 v30; // [rsp+28h] [rbp-98h]
  __int64 v31; // [rsp+30h] [rbp-90h]
  __int64 *v32; // [rsp+38h] [rbp-88h]
  __int64 *v33; // [rsp+40h] [rbp-80h]
  __int64 v34; // [rsp+48h] [rbp-78h]
  __int64 v35; // [rsp+58h] [rbp-68h]
  __int64 v36; // [rsp+58h] [rbp-68h]
  __m128i v37; // [rsp+60h] [rbp-60h] BYREF
  __int64 v38; // [rsp+70h] [rbp-50h]
  __int64 v39; // [rsp+78h] [rbp-48h]
  __int64 v40; // [rsp+80h] [rbp-40h]

  result = *(__int64 **)(a5 + 32);
  v32 = result;
  v27 = &result[*(unsigned int *)(a5 + 40)];
  if ( result != v27 )
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(*v32 + 56);
      v30 = *v32;
      v31 = *v32 + 48;
      if ( v7 != v31 )
        break;
LABEL_4:
      result = ++v32;
      if ( v27 == v32 )
        return result;
    }
    while ( 1 )
    {
      if ( *(_WORD *)(v7 + 68) && *(_WORD *)(v7 + 68) != 68 )
        goto LABEL_4;
      v8 = *(_QWORD *)(v7 + 32);
      v9 = *(_DWORD *)(v7 + 40) & 0xFFFFFF;
      v34 = *(_QWORD *)(a2 + 32);
      if ( v9 == 1 )
      {
        v11 = 0;
        v12 = *(_DWORD *)(v8 + 8);
        if ( !a3 )
          goto LABEL_12;
        i = 0;
        v11 = -1;
        while ( 1 )
        {
LABEL_36:
          v20 = v11 + 1;
          if ( a2 == *(_QWORD *)(v8 + 40 * v20 + 24) )
          {
            sub_2E8A650(v7, v20);
            sub_2E8A650(v7, v11);
          }
          v11 -= 2;
          if ( i == v11 )
            break;
          v8 = *(_QWORD *)(v7 + 32);
        }
        goto LABEL_12;
      }
      for ( i = 1; i != v9; i += 2 )
      {
        if ( *(_QWORD *)(v8 + 40LL * (i + 1) + 24) == a2 )
        {
          v11 = 0;
          v12 = *(_DWORD *)(v8 + 40LL * i + 8);
          if ( a3 )
            goto LABEL_32;
          goto LABEL_12;
        }
      }
      i = 0;
      v11 = 0;
      v12 = *(_DWORD *)(v8 + 8);
      if ( !a3 )
        goto LABEL_12;
LABEL_32:
      v11 = v9 - 2;
      if ( i != v9 - 2 )
        goto LABEL_36;
LABEL_12:
      v13 = *(unsigned int *)(a1 + 168);
      v14 = *(_QWORD *)(a1 + 152);
      if ( (_DWORD)v13 )
      {
        v15 = (v13 - 1) & (37 * v12);
        v16 = (int *)(v14 + 32LL * v15);
        v17 = *v16;
        if ( *v16 == v12 )
        {
LABEL_14:
          if ( v16 != (int *)(v14 + 32 * v13) )
          {
            v35 = *((_QWORD *)v16 + 2);
            if ( v35 != *((_QWORD *)v16 + 1) )
            {
              v18 = *((_QWORD *)v16 + 1);
              do
              {
                v19 = *(_QWORD *)v18;
                if ( sub_2E322C0(*(_QWORD *)v18, v30) )
                {
                  if ( v11 )
                  {
                    sub_2EAB0C0(*(_QWORD *)(v7 + 32) + 40LL * v11, *(_DWORD *)(v18 + 8));
                    *(_QWORD *)(*(_QWORD *)(v7 + 32) + 40LL * (v11 + 1) + 24) = v19;
                  }
                  else
                  {
                    v37.m128i_i32[2] = *(_DWORD *)(v18 + 8);
                    v37.m128i_i64[0] = 0;
                    v38 = 0;
                    v39 = 0;
                    v40 = 0;
                    sub_2E8EAD0(v7, v34, &v37);
                    v37.m128i_i8[0] = 4;
                    v38 = 0;
                    v37.m128i_i32[0] &= 0xFFF000FF;
                    v39 = v19;
                    sub_2E8EAD0(v7, v34, &v37);
                  }
                  v11 = 0;
                }
                v18 += 16;
              }
              while ( v35 != v18 );
            }
LABEL_23:
            if ( v11 )
            {
              sub_2E8A650(v7, v11 + 1);
              sub_2E8A650(v7, v11);
            }
            goto LABEL_25;
          }
        }
        else
        {
          v21 = 1;
          while ( v17 != -1 )
          {
            v25 = v21 + 1;
            v15 = (v13 - 1) & (v21 + v15);
            v16 = (int *)(v14 + 32LL * v15);
            v17 = *v16;
            if ( *v16 == v12 )
              goto LABEL_14;
            v21 = v25;
          }
        }
      }
      v36 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
      if ( v36 == *(_QWORD *)a4 )
        goto LABEL_23;
      v22 = **(_QWORD **)a4;
      if ( !v11 )
        break;
      v33 = *(__int64 **)a4;
      sub_2EAB0C0(*(_QWORD *)(v7 + 32) + 40LL * v11, v12);
      v23 = v33;
      *(_QWORD *)(*(_QWORD *)(v7 + 32) + 40LL * (v11 + 1) + 24) = v22;
      for ( j = v33 + 1; j != (__int64 *)v36; ++j )
      {
        v22 = v23[1];
LABEL_44:
        v37.m128i_i32[2] = v12;
        v37.m128i_i64[0] = 0;
        v38 = 0;
        v39 = 0;
        v40 = 0;
        sub_2E8EAD0(v7, v34, &v37);
        v37.m128i_i8[0] = 4;
        v38 = 0;
        v37.m128i_i32[0] &= 0xFFF000FF;
        v39 = v22;
        sub_2E8EAD0(v7, v34, &v37);
        v23 = j;
      }
LABEL_25:
      if ( (*(_BYTE *)v7 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v7 + 44) & 8) != 0 )
          v7 = *(_QWORD *)(v7 + 8);
      }
      v7 = *(_QWORD *)(v7 + 8);
      if ( v31 == v7 )
        goto LABEL_4;
    }
    j = *(__int64 **)a4;
    goto LABEL_44;
  }
  return result;
}

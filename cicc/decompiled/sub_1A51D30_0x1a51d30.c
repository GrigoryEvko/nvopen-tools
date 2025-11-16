// Function: sub_1A51D30
// Address: 0x1a51d30
//
char *__fastcall sub_1A51D30(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, char *a5, __int64 a6)
{
  char *result; // rax
  _QWORD *v9; // r15
  char *v10; // r12
  _QWORD *v11; // rbx
  bool v12; // zf
  __int64 v13; // rax
  __int64 v14; // rbx
  char *v15; // r15
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rbx
  char *v19; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdi
  char *v24; // rdx
  __int64 i; // rcx
  __int64 v26; // r8
  __int64 v27; // rdx
  char *v28; // r8
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // r9
  size_t v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // r13
  char *v38; // r14
  char *dest; // [rsp+8h] [rbp-48h]

  result = (char *)a1;
  if ( a4 != 1 )
  {
    if ( a4 > a6 )
    {
      v14 = a4 / 2;
      v15 = (char *)&a1[a4 / 2];
      v16 = sub_1A51D30(a1, &a1[v14], a3, v14, a5, a6);
      v17 = a4 - v14;
      v18 = (__int64)&a1[v14];
      dest = (char *)v16;
      if ( v17 )
      {
        while ( sub_1377F70(a3, **(_QWORD **)(*(_QWORD *)v18 + 32LL)) )
        {
          v18 += 8;
          if ( !--v17 )
            goto LABEL_18;
        }
        v18 = sub_1A51D30(v18, a2, a3, v17, a5, a6);
      }
LABEL_18:
      v19 = dest;
      if ( dest != v15 )
      {
        if ( v15 == (char *)v18 )
        {
          return dest;
        }
        else
        {
          v20 = v18;
          v18 = (__int64)&dest[v18 - (_QWORD)v15];
          v21 = (v20 - (__int64)dest) >> 3;
          v22 = (v15 - dest) >> 3;
          if ( v22 == v21 - v22 )
          {
            v33 = 0;
            do
            {
              v34 = *(_QWORD *)&dest[v33];
              *(_QWORD *)&dest[v33] = *(_QWORD *)&v15[v33];
              *(_QWORD *)&v15[v33] = v34;
              v33 += 8;
            }
            while ( v15 - dest != v33 );
            return v15;
          }
          else
          {
            while ( 1 )
            {
              v23 = v21 - v22;
              if ( v22 < v21 - v22 )
                break;
LABEL_29:
              v28 = &v19[8 * v21];
              if ( v23 == 1 )
              {
                v35 = *((_QWORD *)v28 - 1);
                if ( v19 != v28 - 8 )
                  memmove(v19 + 8, v19, 8 * v21 - 8);
                *(_QWORD *)v19 = v35;
                return (char *)v18;
              }
              v19 = &v28[-8 * v23];
              if ( v22 > 0 )
              {
                v29 = -8;
                v30 = 0;
                do
                {
                  v31 = *(_QWORD *)&v19[v29];
                  ++v30;
                  *(_QWORD *)&v19[v29] = *(_QWORD *)&v28[v29];
                  *(_QWORD *)&v28[v29] = v31;
                  v29 -= 8;
                }
                while ( v22 != v30 );
                v19 -= 8 * v22;
              }
              v22 = v21 % v23;
              if ( !(v21 % v23) )
                return (char *)v18;
              v21 = v23;
            }
            while ( v22 != 1 )
            {
              v24 = &v19[8 * v22];
              if ( v23 > 0 )
              {
                for ( i = 0; i != v23; ++i )
                {
                  v26 = *(_QWORD *)&v19[8 * i];
                  *(_QWORD *)&v19[8 * i] = *(_QWORD *)&v24[8 * i];
                  *(_QWORD *)&v24[8 * i] = v26;
                }
                v19 += 8 * v23;
              }
              v27 = v21 % v22;
              if ( !(v21 % v22) )
                return (char *)v18;
              v21 = v22;
              v22 -= v27;
              v23 = v21 - v22;
              if ( v22 >= v21 - v22 )
                goto LABEL_29;
            }
            v36 = 8 * v21;
            v37 = *(_QWORD *)v19;
            v38 = &v19[v36];
            if ( &v19[v36] != v19 + 8 )
              memmove(v19, v19 + 8, v36 - 8);
            *((_QWORD *)v38 - 1) = v37;
          }
        }
      }
      return (char *)v18;
    }
    else
    {
      v9 = a1;
      *(_QWORD *)a5 = *a1;
      v10 = a5 + 8;
      v11 = a1 + 1;
      if ( a1 + 1 == a2 )
      {
        v32 = 8;
      }
      else
      {
        do
        {
          while ( 1 )
          {
            v12 = !sub_1377F70(a3, **(_QWORD **)(*v11 + 32LL));
            v13 = *v11;
            if ( v12 )
              break;
            ++v11;
            *v9++ = v13;
            if ( a2 == v11 )
              goto LABEL_8;
          }
          ++v11;
          *(_QWORD *)v10 = v13;
          v10 += 8;
        }
        while ( a2 != v11 );
LABEL_8:
        v32 = v10 - a5;
      }
      if ( a5 != v10 )
        memmove(v9, a5, v32);
      return (char *)v9;
    }
  }
  return result;
}

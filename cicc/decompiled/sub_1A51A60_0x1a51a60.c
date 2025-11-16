// Function: sub_1A51A60
// Address: 0x1a51a60
//
__int64 *__fastcall sub_1A51A60(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 *result; // rax
  __int64 *v9; // r15
  __int64 *v10; // r12
  __int64 *v11; // rbx
  bool v12; // zf
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 *v18; // rbx
  __int64 *v19; // r12
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 *v24; // rdx
  __int64 j; // rcx
  __int64 v26; // r8
  __int64 v27; // rdx
  __int64 *v28; // r8
  __int64 v29; // rcx
  __int64 i; // rdx
  __int64 v31; // r9
  size_t v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 *v38; // r14
  __int64 *dest; // [rsp+8h] [rbp-48h]

  result = a1;
  if ( a4 != 1 )
  {
    if ( a4 > a6 )
    {
      v14 = a4 / 2;
      v15 = &a1[a4 / 2];
      v16 = sub_1A51A60(a1, &a1[v14], a3, v14, a5, a6);
      v17 = a4 - v14;
      v18 = &a1[v14];
      dest = (__int64 *)v16;
      if ( v17 )
      {
        while ( sub_1377F70(a3, *v18) )
        {
          ++v18;
          if ( !--v17 )
            goto LABEL_18;
        }
        v18 = (__int64 *)sub_1A51A60(v18, a2, a3, v17, a5, a6);
      }
LABEL_18:
      v19 = dest;
      if ( dest != v15 )
      {
        if ( v15 == v18 )
        {
          return dest;
        }
        else
        {
          v20 = v18;
          v18 = (__int64 *)((char *)dest + (char *)v18 - (char *)v15);
          v21 = v20 - dest;
          v22 = v15 - dest;
          if ( v22 == v21 - v22 )
          {
            v33 = 0;
            do
            {
              v34 = dest[v33 / 8];
              dest[v33 / 8] = v15[v33 / 8];
              v15[v33 / 8] = v34;
              v33 += 8LL;
            }
            while ( v33 != (char *)v15 - (char *)dest );
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
              v28 = &v19[v21];
              if ( v23 == 1 )
              {
                v35 = *(v28 - 1);
                if ( v19 != v28 - 1 )
                  memmove(v19 + 1, v19, 8 * v21 - 8);
                *v19 = v35;
                return v18;
              }
              v19 = &v28[-v23];
              if ( v22 > 0 )
              {
                v29 = 0x1FFFFFFFFFFFFFFFLL;
                for ( i = 0; i != v22; ++i )
                {
                  v31 = v19[v29];
                  v19[v29] = v28[v29];
                  v28[v29--] = v31;
                }
                v19 -= v22;
              }
              v22 = v21 % v23;
              if ( !(v21 % v23) )
                return v18;
              v21 = v23;
            }
            while ( v22 != 1 )
            {
              v24 = &v19[v22];
              if ( v23 > 0 )
              {
                for ( j = 0; j != v23; ++j )
                {
                  v26 = v19[j];
                  v19[j] = v24[j];
                  v24[j] = v26;
                }
                v19 += v23;
              }
              v27 = v21 % v22;
              if ( !(v21 % v22) )
                return v18;
              v21 = v22;
              v22 -= v27;
              v23 = v21 - v22;
              if ( v22 >= v21 - v22 )
                goto LABEL_29;
            }
            v36 = v21;
            v37 = *v19;
            v38 = &v19[v36];
            if ( &v19[v36] != v19 + 1 )
              memmove(v19, v19 + 1, v36 * 8 - 8);
            *(v38 - 1) = v37;
          }
        }
      }
      return v18;
    }
    else
    {
      v9 = a1;
      *a5 = *a1;
      v10 = a5 + 1;
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
            v12 = !sub_1377F70(a3, *v11);
            v13 = *v11;
            if ( v12 )
              break;
            ++v11;
            *v9++ = v13;
            if ( a2 == v11 )
              goto LABEL_8;
          }
          ++v11;
          *v10++ = v13;
        }
        while ( a2 != v11 );
LABEL_8:
        v32 = (char *)v10 - (char *)a5;
      }
      if ( a5 != v10 )
        memmove(v9, a5, v32);
      return v9;
    }
  }
  return result;
}

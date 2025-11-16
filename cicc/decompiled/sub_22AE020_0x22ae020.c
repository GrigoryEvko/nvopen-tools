// Function: sub_22AE020
// Address: 0x22ae020
//
char **__fastcall sub_22AE020(char **a1, char **a2, char **a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **result; // rax
  __int64 v7; // r13
  __int64 v8; // r15
  char **v9; // r12
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // rdx
  char **v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  char **v19; // r15
  __int64 i; // r13
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rcx
  char **v26; // r12
  char **v27; // r15
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  char *v35; // [rsp+8h] [rbp-A8h]
  char **v36; // [rsp+10h] [rbp-A0h]
  signed __int64 v37; // [rsp+18h] [rbp-98h]
  signed __int64 v38; // [rsp+20h] [rbp-90h]
  __int64 v39; // [rsp+28h] [rbp-88h]
  unsigned int v40; // [rsp+30h] [rbp-80h]
  unsigned int v41; // [rsp+30h] [rbp-80h]
  unsigned int v42; // [rsp+30h] [rbp-80h]
  char *v43; // [rsp+38h] [rbp-78h] BYREF
  __int64 v44; // [rsp+40h] [rbp-70h]
  _BYTE v45[104]; // [rsp+48h] [rbp-68h] BYREF

  result = a3;
  v36 = a1;
  if ( a1 != a2 )
  {
    result = a1;
    if ( a2 != a3 )
    {
      v35 = (char *)a1 + (char *)a3 - (char *)a2;
      v37 = 0x8E38E38E38E38E39LL * (a3 - a1);
      v39 = 0x8E38E38E38E38E39LL * (a2 - a1);
      if ( v39 != v37 - v39 )
      {
        while ( 1 )
        {
          v38 = v37 - v39;
          if ( v39 >= v37 - v39 )
          {
            v16 = &v36[9 * v37];
            v17 = 72 * v38;
            v36 = &v16[-9 * v38];
            if ( v39 > 0 )
            {
              v18 = (__int64)&v16[-9 * v38 - 8];
              v19 = v16 - 8;
              for ( i = 0; i != v39; ++i )
              {
                v25 = *(unsigned int *)(v18 - 8);
                v43 = v45;
                v44 = 0xC00000000LL;
                v41 = v25;
                if ( *(_DWORD *)(v18 + 8) )
                  sub_22AD4A0((__int64)&v43, (char **)v18, v17, v25, a5, a6);
                v21 = *((unsigned int *)v19 - 2);
                *(_DWORD *)(v18 - 8) = v21;
                sub_22AD4A0(v18, v19, v17, v21, a5, a6);
                *((_DWORD *)v19 - 2) = v41;
                sub_22AD4A0((__int64)v19, &v43, v22, v41, v23, v24);
                if ( v43 != v45 )
                  _libc_free((unsigned __int64)v43);
                v18 -= 72;
                v19 -= 9;
              }
              v36 -= 9 * v39;
            }
            v39 = v37 % v38;
            if ( !(v37 % v38) )
              return (char **)v35;
          }
          else
          {
            if ( v37 - v39 > 0 )
            {
              v7 = 0;
              v8 = (__int64)(v36 + 1);
              v9 = &v36[9 * v39 + 1];
              do
              {
                v14 = *(unsigned int *)(v8 - 8);
                v15 = *(unsigned int *)(v8 + 8);
                v43 = v45;
                v44 = 0xC00000000LL;
                v40 = v14;
                if ( (_DWORD)v15 )
                  sub_22AD4A0((__int64)&v43, (char **)v8, v15, v14, a5, a6);
                v10 = *((unsigned int *)v9 - 2);
                *(_DWORD *)(v8 - 8) = v10;
                sub_22AD4A0(v8, v9, v15, v10, a5, a6);
                *((_DWORD *)v9 - 2) = v40;
                sub_22AD4A0((__int64)v9, &v43, v11, v40, v12, v13);
                if ( v43 != v45 )
                  _libc_free((unsigned __int64)v43);
                ++v7;
                v8 += 72;
                v9 += 9;
              }
              while ( v38 != v7 );
              v36 += 9 * v38;
            }
            if ( !(v37 % v39) )
              return (char **)v35;
            v38 = v39;
            v39 -= v37 % v39;
          }
          v37 = v38;
        }
      }
      v26 = a1;
      v27 = a2 + 1;
      do
      {
        v32 = *(unsigned int *)v26;
        v33 = *((unsigned int *)v26 + 4);
        v43 = v45;
        v44 = 0xC00000000LL;
        v34 = (__int64)(v26 + 1);
        v42 = v32;
        if ( (_DWORD)v33 )
        {
          sub_22AD4A0((__int64)&v43, v26 + 1, v32, v33, v34, a6);
          v34 = (__int64)(v26 + 1);
        }
        v28 = *((unsigned int *)v27 - 2);
        *(_DWORD *)v26 = v28;
        sub_22AD4A0(v34, v27, v28, v33, v34, a6);
        *((_DWORD *)v27 - 2) = v42;
        sub_22AD4A0((__int64)v27, &v43, v42, v29, v30, v31);
        if ( v43 != v45 )
          _libc_free((unsigned __int64)v43);
        v26 += 9;
        v27 += 9;
      }
      while ( a2 != v26 );
      return a2;
    }
  }
  return result;
}

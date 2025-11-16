// Function: sub_D6DC00
// Address: 0xd6dc00
//
unsigned __int64 __fastcall sub_D6DC00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  unsigned __int64 result; // rax
  __int64 v8; // rsi
  unsigned int v10; // r9d
  __int64 *v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r8
  __int64 v14; // rcx
  int v15; // r9d
  __int64 v16; // r15
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 *v19; // rax
  __int64 v20; // rbx
  __int64 j; // rbx
  unsigned int v22; // r8d
  __int64 *v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rcx
  int v29; // eax
  __int64 v30; // rsi
  int i; // edx
  int v32; // edx
  int v33; // r10d
  int v34; // r11d
  int v35; // [rsp+4h] [rbp-3Ch]
  unsigned int v36; // [rsp+8h] [rbp-38h]
  int v37; // [rsp+Ch] [rbp-34h]

  v6 = *(_QWORD *)a1;
  result = *(unsigned int *)(v6 + 88);
  v8 = *(_QWORD *)(v6 + 72);
  if ( (_DWORD)result )
  {
    v36 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v10 = (result - 1) & v36;
    v11 = (__int64 *)(v8 + 16LL * v10);
    v12 = *v11;
    if ( a2 != *v11 )
    {
      for ( i = 1; ; i = v34 )
      {
        if ( v12 == -4096 )
          return result;
        v34 = i + 1;
        v10 = (result - 1) & (i + v10);
        v11 = (__int64 *)(v8 + 16LL * v10);
        v12 = *v11;
        if ( a2 == *v11 )
          break;
      }
    }
    result = v8 + 16 * result;
    if ( v11 != (__int64 *)result )
    {
      v13 = v11[1];
      if ( v13 )
      {
        v14 = a4 + 24;
        if ( v14 == a3 + 48 )
          goto LABEL_26;
        v15 = *(_DWORD *)(v6 + 56);
        v16 = *(_QWORD *)(v6 + 40);
        v37 = v15 - 1;
        while ( 1 )
        {
          v17 = v14 - 24;
          if ( !v14 )
            v17 = 0;
          if ( v15 )
          {
            v18 = v37 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
            v19 = (__int64 *)(v16 + 16LL * v18);
            v20 = *v19;
            if ( v17 == *v19 )
            {
LABEL_11:
              j = v19[1];
              if ( j )
              {
LABEL_16:
                v25 = *(_QWORD *)(j + 40);
                if ( v13 )
                {
                  if ( v13 == v25 || (v13 = v25 - 32, !v25) )
                    v13 = 0;
                }
                v26 = j;
                for ( j = v13; ; j = 0 )
                {
                  sub_1041EA0(v6, v26, a3, 1);
                  v6 = *(_QWORD *)a1;
                  v27 = *(unsigned int *)(*(_QWORD *)a1 + 88LL);
                  v28 = *(_QWORD *)(*(_QWORD *)a1 + 72LL);
                  if ( (_DWORD)v27 )
                  {
                    v22 = (v27 - 1) & v36;
                    v23 = (__int64 *)(v28 + 16LL * v22);
                    v24 = *v23;
                    if ( a2 == *v23 )
                    {
LABEL_14:
                      if ( v23 != (__int64 *)(v28 + 16 * v27) )
                      {
                        v13 = v23[1];
                        if ( j )
                          goto LABEL_16;
LABEL_26:
                        result = sub_D68C20(v6, a2);
                        if ( result && result != (*(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL) )
                        {
                          v30 = *(_QWORD *)(result + 8);
                          if ( !v30 )
                            BUG();
                          if ( *(_BYTE *)(v30 - 48) == 28 )
                            return sub_D6D630(a1, v30 - 48);
                        }
                        return result;
                      }
                    }
                    else
                    {
                      v32 = 1;
                      while ( v24 != -4096 )
                      {
                        v33 = v32 + 1;
                        v22 = (v27 - 1) & (v32 + v22);
                        v23 = (__int64 *)(v28 + 16LL * v22);
                        v24 = *v23;
                        if ( a2 == *v23 )
                          goto LABEL_14;
                        v32 = v33;
                      }
                    }
                  }
                  v26 = j;
                  if ( !j )
                    goto LABEL_26;
                }
              }
            }
            else
            {
              v29 = 1;
              while ( v20 != -4096 )
              {
                v18 = v37 & (v29 + v18);
                v35 = v29 + 1;
                v19 = (__int64 *)(v16 + 16LL * v18);
                v20 = *v19;
                if ( v17 == *v19 )
                  goto LABEL_11;
                v29 = v35;
              }
            }
          }
          v14 = *(_QWORD *)(v14 + 8);
          if ( a3 + 48 == v14 )
            goto LABEL_26;
        }
      }
    }
  }
  return result;
}

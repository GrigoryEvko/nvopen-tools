// Function: sub_EA3570
// Address: 0xea3570
//
unsigned __int64 __fastcall sub_EA3570(int **a1, unsigned int a2)
{
  int **v2; // r13
  int v3; // edx
  __int64 v4; // rax
  unsigned int *v5; // r14
  unsigned int *v6; // rbx
  unsigned __int64 result; // rax
  __int64 v8; // r12
  _BYTE *v9; // rsi
  void *v10; // rdi
  size_t v11; // r15
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rsi
  size_t v14; // r11
  unsigned __int8 *v15; // rsi
  _QWORD *v16; // r10
  _BYTE *v17; // r15
  _QWORD *v18; // rdx
  size_t v19; // r14
  size_t v20; // r12
  size_t v21; // r13
  char v22; // al
  unsigned __int64 v23; // rcx
  int **v24; // [rsp+8h] [rbp-88h]
  __int64 v25; // [rsp+10h] [rbp-80h]
  unsigned int *v26; // [rsp+20h] [rbp-70h]
  _QWORD *v27; // [rsp+28h] [rbp-68h]
  char v28; // [rsp+3Eh] [rbp-52h]
  char v29; // [rsp+3Fh] [rbp-51h]
  _QWORD *v30; // [rsp+40h] [rbp-50h] BYREF
  size_t v31; // [rsp+48h] [rbp-48h]
  _QWORD v32[8]; // [rsp+50h] [rbp-40h] BYREF

  v2 = a1;
  v29 = 0;
  v3 = **a1;
  if ( v3 )
  {
    v29 = *(_BYTE *)(*(_QWORD *)a1[1] + 48LL * *((_QWORD *)a1[1] + 1) - 7);
    if ( v29 )
      v29 = v3 - 1 == a2;
  }
  v4 = *(_QWORD *)a1[2] + 24LL * a2;
  v5 = *(unsigned int **)(v4 + 8);
  v6 = *(unsigned int **)v4;
  result = (unsigned __int64)v32;
  if ( v6 != v5 )
  {
    while ( 1 )
    {
      v8 = (__int64)v2[4];
      v9 = (_BYTE *)*((_QWORD *)v6 + 1);
      result = *v6;
      if ( !*((_BYTE *)v2[3] + 871) )
        goto LABEL_11;
      if ( *v9 == 37 )
      {
        if ( (_DWORD)result == 4 )
        {
          v13 = (_QWORD *)*((_QWORD *)v6 + 3);
          if ( v6[8] > 0x40 )
            v13 = (_QWORD *)*v13;
          result = sub_CB59F0((__int64)v2[4], (signed __int64)v13);
          goto LABEL_7;
        }
LABEL_11:
        if ( (_DWORD)result != 3 || v29 )
          goto LABEL_13;
        v12 = *((_QWORD *)v6 + 2);
        if ( v12 >= 2 )
        {
          v10 = *(void **)(v8 + 32);
          v11 = v12 - 2;
          ++v9;
          result = *(_QWORD *)(v8 + 24) - (_QWORD)v10;
          if ( result < v12 - 2 )
            goto LABEL_14;
          if ( v12 != 2 )
          {
LABEL_19:
            result = (unsigned __int64)memcpy(v10, v9, v11);
            *(_QWORD *)(v8 + 32) += v11;
          }
        }
LABEL_7:
        v6 += 10;
        if ( v5 == v6 )
          return result;
      }
      else
      {
        if ( *v9 != 60 )
          goto LABEL_11;
        if ( (_DWORD)result == 3 )
        {
          v14 = *((_QWORD *)v6 + 2);
          if ( v14 )
          {
            if ( v14 == 1 )
            {
              v15 = (unsigned __int8 *)v32;
              v31 = 0;
              v14 = 0;
              LOBYTE(v32[0]) = 0;
              v30 = v32;
            }
            else
            {
              v16 = v32;
              v31 = 0;
              LOBYTE(v32[0]) = 0;
              v30 = v32;
              v14 -= 2LL;
              if ( v14 )
              {
                v26 = v5;
                v18 = v32;
                v19 = 0;
                v17 = v9 + 1;
                v25 = v8;
                v20 = 0;
                v24 = v2;
                v21 = v14;
                while ( 1 )
                {
                  v22 = v17[v19];
                  if ( v22 == 33 )
                    v22 = v17[++v19];
                  v23 = 15;
                  if ( v18 != v16 )
                    v23 = v32[0];
                  if ( v20 + 1 > v23 )
                  {
                    v27 = v16;
                    v28 = v22;
                    sub_2240BB0(&v30, v20, 0, 0, 1);
                    v18 = v30;
                    v16 = v27;
                    v22 = v28;
                  }
                  *((_BYTE *)v18 + v20) = v22;
                  ++v19;
                  v31 = v20 + 1;
                  *((_BYTE *)v30 + v20 + 1) = 0;
                  if ( v19 >= v21 )
                    break;
                  v20 = v31;
                  v18 = v30;
                }
                v5 = v26;
                v8 = v25;
                v2 = v24;
                v14 = v31;
                v15 = (unsigned __int8 *)v30;
              }
              else
              {
                v15 = (unsigned __int8 *)v32;
              }
            }
          }
          else
          {
            v31 = 0;
            LOBYTE(v32[0]) = 0;
            v30 = v32;
            v15 = (unsigned __int8 *)v32;
          }
          result = sub_CB6200(v8, v15, v14);
          if ( v30 != v32 )
            result = j_j___libc_free_0(v30, v32[0] + 1LL);
          goto LABEL_7;
        }
LABEL_13:
        v10 = *(void **)(v8 + 32);
        v11 = *((_QWORD *)v6 + 2);
        result = *(_QWORD *)(v8 + 24) - (_QWORD)v10;
        if ( v11 <= result )
        {
          if ( v11 )
            goto LABEL_19;
          goto LABEL_7;
        }
LABEL_14:
        v6 += 10;
        result = sub_CB6200((__int64)v2[4], v9, v11);
        if ( v5 == v6 )
          return result;
      }
    }
  }
  return result;
}

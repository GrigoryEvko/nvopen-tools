// Function: sub_AE6050
// Address: 0xae6050
//
__int64 __fastcall sub_AE6050(_QWORD *a1, _BYTE *a2)
{
  __int64 result; // rax
  __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rdi
  __int64 v8; // rcx
  _QWORD *v9; // rdx
  __int64 v10; // r15
  _BYTE *v11; // rsi
  _QWORD *v12; // r12
  _BYTE *v13; // r14
  __int64 v14; // rdi
  _BYTE *v15; // rbx
  _QWORD *v16; // rax
  __int64 v17; // rcx
  _QWORD *v18; // rdx
  char v19; // dl
  char v20; // dl
  __int64 v21; // r15
  __int64 v22; // rax
  _QWORD *v23; // [rsp+0h] [rbp-70h] BYREF
  int v24; // [rsp+8h] [rbp-68h]
  _BYTE v25[96]; // [rsp+10h] [rbp-60h] BYREF

  result = sub_B9F8A0(*a1);
  if ( result )
  {
    v5 = *(_QWORD *)(result + 16);
    if ( v5 )
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(v5 + 24);
        if ( *(_BYTE *)v6 != 85 )
          goto LABEL_4;
        result = *(_QWORD *)(v6 - 32);
        if ( !result
          || *(_BYTE *)result
          || *(_QWORD *)(result + 24) != *(_QWORD *)(v6 + 80)
          || (*(_BYTE *)(result + 33) & 0x20) == 0 )
        {
          goto LABEL_4;
        }
        result = *(unsigned int *)(result + 36);
        if ( (unsigned int)result > 0x45 )
        {
          if ( (_DWORD)result != 71 )
            goto LABEL_4;
          v7 = a1[1];
          if ( !*(_BYTE *)(v7 + 28) )
            goto LABEL_36;
LABEL_13:
          result = *(_QWORD *)(v7 + 8);
          v8 = *(unsigned int *)(v7 + 20);
          v9 = (_QWORD *)(result + 8 * v8);
          if ( (_QWORD *)result != v9 )
          {
            while ( v6 != *(_QWORD *)result )
            {
              result += 8;
              if ( v9 == (_QWORD *)result )
                goto LABEL_16;
            }
            goto LABEL_4;
          }
LABEL_16:
          if ( (unsigned int)v8 >= *(_DWORD *)(v7 + 16) )
            goto LABEL_36;
          *(_DWORD *)(v7 + 20) = v8 + 1;
          *v9 = v6;
          ++*(_QWORD *)v7;
LABEL_18:
          v10 = a1[3];
          result = *(unsigned int *)(v10 + 8);
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(v10 + 12) )
          {
            sub_C8D5F0(a1[3], v10 + 16, result + 1, 8);
            result = *(unsigned int *)(v10 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v10 + 8 * result) = v6;
          ++*(_DWORD *)(v10 + 8);
          v5 = *(_QWORD *)(v5 + 8);
          if ( !v5 )
            break;
        }
        else
        {
          if ( (unsigned int)result <= 0x43 )
            goto LABEL_4;
          v7 = a1[1];
          if ( *(_BYTE *)(v7 + 28) )
            goto LABEL_13;
LABEL_36:
          result = sub_C8CC70(v7, *(_QWORD *)(v5 + 24));
          if ( v19 )
            goto LABEL_18;
LABEL_4:
          v5 = *(_QWORD *)(v5 + 8);
          if ( !v5 )
            break;
        }
      }
    }
  }
  if ( a1[4] && *a2 == 2 )
  {
    v11 = a2 + 8;
    sub_B967C0(&v23, a2 + 8);
    v12 = v23;
    v13 = &v23[v24];
    if ( v23 != (_QWORD *)v13 )
    {
      while ( 1 )
      {
        v14 = a1[2];
        v15 = (_BYTE *)*v12;
        if ( *(_BYTE *)(v14 + 28) )
        {
          v16 = *(_QWORD **)(v14 + 8);
          v17 = *(unsigned int *)(v14 + 20);
          v18 = &v16[v17];
          if ( v16 != v18 )
          {
            while ( v15 != (_BYTE *)*v16 )
            {
              if ( v18 == ++v16 )
                goto LABEL_43;
            }
            goto LABEL_29;
          }
LABEL_43:
          if ( (unsigned int)v17 < *(_DWORD *)(v14 + 16) )
          {
            *(_DWORD *)(v14 + 20) = v17 + 1;
            *v18 = v15;
            ++*(_QWORD *)v14;
            goto LABEL_39;
          }
        }
        v11 = (_BYTE *)*v12;
        sub_C8CC70(v14, *v12);
        if ( v20 )
        {
LABEL_39:
          v21 = a1[4];
          v22 = *(unsigned int *)(v21 + 8);
          if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
          {
            v11 = (_BYTE *)(v21 + 16);
            sub_C8D5F0(a1[4], v21 + 16, v22 + 1, 8);
            v22 = *(unsigned int *)(v21 + 8);
          }
          ++v12;
          *(_QWORD *)(*(_QWORD *)v21 + 8 * v22) = v15;
          ++*(_DWORD *)(v21 + 8);
          if ( v13 == (_BYTE *)v12 )
          {
LABEL_30:
            v13 = v23;
            break;
          }
        }
        else
        {
LABEL_29:
          if ( v13 == (_BYTE *)++v12 )
            goto LABEL_30;
        }
      }
    }
    result = (__int64)v25;
    if ( v13 != v25 )
      return _libc_free(v13, v11);
  }
  return result;
}

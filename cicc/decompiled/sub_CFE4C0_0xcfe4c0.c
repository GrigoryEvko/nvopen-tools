// Function: sub_CFE4C0
// Address: 0xcfe4c0
//
__int64 __fastcall sub_CFE4C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 *v6; // rdx
  __int64 result; // rax
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // [rsp+0h] [rbp-80h]
  __int64 v18; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+10h] [rbp-70h] BYREF
  __int64 *v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+20h] [rbp-60h]
  int v22; // [rsp+28h] [rbp-58h]
  char v23; // [rsp+2Ch] [rbp-54h]
  _BYTE v24[80]; // [rsp+30h] [rbp-50h] BYREF

  v6 = (__int64 *)v24;
  result = *(unsigned int *)(a1 + 192);
  v20 = (__int64 *)v24;
  v19 = 0;
  v21 = 4;
  v22 = 0;
  v23 = 1;
  if ( (_DWORD)result )
  {
    v8 = *(_QWORD *)(a1 + 184);
    result = v8 + 48LL * *(unsigned int *)(a1 + 200);
    v17 = result;
    if ( v8 != result )
    {
      while ( 1 )
      {
        result = *(_QWORD *)(v8 + 24);
        if ( result != -4096 && result != -8192 )
          break;
        v8 += 48;
        if ( v17 == v8 )
          return result;
      }
      if ( v17 != v8 )
      {
        v9 = *(_QWORD *)(v8 + 40);
        if ( !*(_BYTE *)(v9 + 192) )
          goto LABEL_50;
        while ( 1 )
        {
          v10 = *(_QWORD *)(v9 + 16);
          v11 = v10 + 32LL * *(unsigned int *)(v9 + 24);
          if ( v11 != v10 )
          {
            while ( 1 )
            {
              while ( 1 )
              {
                a2 = *(_QWORD *)(v10 + 16);
                if ( a2 )
                  break;
LABEL_17:
                v10 += 32;
                if ( v11 == v10 )
                  goto LABEL_18;
              }
              if ( v23 )
              {
                v12 = v20;
                v6 = &v20[HIDWORD(v21)];
                if ( v20 != v6 )
                {
                  while ( a2 != *v12 )
                  {
                    if ( v6 == ++v12 )
                      goto LABEL_45;
                  }
                  goto LABEL_17;
                }
LABEL_45:
                if ( HIDWORD(v21) >= (unsigned int)v21 )
                  goto LABEL_43;
                v10 += 32;
                ++HIDWORD(v21);
                *v6 = a2;
                ++v19;
                if ( v11 == v10 )
                  break;
              }
              else
              {
LABEL_43:
                v10 += 32;
                sub_C8CC70((__int64)&v19, a2, (__int64)v6, a4, a5, (__int64)a6);
                if ( v11 == v10 )
                  break;
              }
            }
          }
LABEL_18:
          v13 = *(_QWORD *)(v8 + 24);
          v14 = *(_QWORD *)(v13 + 80);
          result = v13 + 72;
          v18 = result;
          if ( v14 != result )
            break;
LABEL_37:
          v8 += 48;
          if ( v8 == v17 )
            goto LABEL_41;
          while ( 1 )
          {
            result = *(_QWORD *)(v8 + 24);
            if ( result != -8192 && result != -4096 )
              break;
            v8 += 48;
            if ( v17 == v8 )
              goto LABEL_41;
          }
          if ( v8 == v17 )
          {
LABEL_41:
            if ( !v23 )
              return _libc_free(v20, a2);
            return result;
          }
          v9 = *(_QWORD *)(v8 + 40);
          if ( !*(_BYTE *)(v9 + 192) )
LABEL_50:
            sub_CFDFC0(v9, a2, (__int64)v6, a4, a5, a6);
        }
        while ( 1 )
        {
          if ( !v14 )
            BUG();
          v15 = *(_QWORD *)(v14 + 32);
          v16 = v14 + 24;
          if ( v15 != v14 + 24 )
            break;
LABEL_36:
          v14 = *(_QWORD *)(v14 + 8);
          if ( v18 == v14 )
            goto LABEL_37;
        }
        while ( 1 )
        {
          if ( !v15 )
            BUG();
          if ( *(_BYTE *)(v15 - 24) != 85 )
            goto LABEL_22;
          result = *(_QWORD *)(v15 - 56);
          if ( !result )
            goto LABEL_22;
          if ( *(_BYTE *)result )
            goto LABEL_22;
          a4 = *(_QWORD *)(v15 + 56);
          if ( *(_QWORD *)(result + 24) != a4 || *(_DWORD *)(result + 36) != 11 )
            goto LABEL_22;
          a2 = v15 - 24;
          if ( v23 )
          {
            result = (__int64)v20;
            v6 = &v20[HIDWORD(v21)];
            if ( v20 == v6 )
              goto LABEL_33;
            while ( a2 != *(_QWORD *)result )
            {
              result += 8;
              if ( v6 == (__int64 *)result )
                goto LABEL_33;
            }
LABEL_22:
            v15 = *(_QWORD *)(v15 + 8);
            if ( v16 == v15 )
              goto LABEL_36;
          }
          else
          {
            result = (__int64)sub_C8CA60((__int64)&v19, a2);
            if ( !result )
LABEL_33:
              sub_C64ED0("Assumption in scanned function not in cache", 1u);
            v15 = *(_QWORD *)(v15 + 8);
            if ( v16 == v15 )
              goto LABEL_36;
          }
        }
      }
    }
  }
  return result;
}

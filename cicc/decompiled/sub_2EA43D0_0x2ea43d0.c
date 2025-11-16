// Function: sub_2EA43D0
// Address: 0x2ea43d0
//
__int64 __fastcall sub_2EA43D0(__int64 a1)
{
  __int64 v2; // rbx
  __int64 *v3; // r14
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 *v7; // r12
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  _QWORD *v10; // rdi
  __int64 v11; // r8
  __int64 *v12; // r13
  _QWORD *v13; // rdi
  __int64 v14; // r8
  _QWORD *v15; // rax
  __int64 v16; // r8
  char v18; // r8
  __int64 v19; // rsi
  _QWORD *v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // rsi
  _QWORD *v23; // rax
  _QWORD *v24; // rdi
  __int64 v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // rdi
  __int64 *v28; // [rsp+8h] [rbp-58h]
  __int64 v29; // [rsp+10h] [rbp-50h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  __int64 *v31; // [rsp+20h] [rbp-40h]
  __int64 *v32; // [rsp+28h] [rbp-38h]

  v2 = a1 + 56;
  v30 = 0;
  v28 = *(__int64 **)(a1 + 40);
  v32 = *(__int64 **)(a1 + 32);
  if ( v32 != v28 )
  {
    while ( 1 )
    {
      v3 = *(__int64 **)(*v32 + 112);
      v29 = *v32;
      v4 = 8LL * *(unsigned int *)(*v32 + 120);
      v31 = &v3[(unsigned __int64)v4 / 8];
      v5 = v4 >> 3;
      v6 = v4 >> 5;
      if ( v6 )
      {
        v7 = &v3[4 * v6];
        while ( !*(_BYTE *)(a1 + 84) )
        {
          if ( !sub_C8CA60(v2, *v3) )
            goto LABEL_29;
          v11 = v3[1];
          v12 = v3 + 1;
          if ( *(_BYTE *)(a1 + 84) )
          {
            v8 = *(_QWORD **)(a1 + 64);
            v9 = &v8[*(unsigned int *)(a1 + 76)];
            if ( v9 != v8 )
            {
              v13 = *(_QWORD **)(a1 + 64);
LABEL_11:
              while ( *v8 != v11 )
              {
                if ( ++v8 == v9 )
                  goto LABEL_28;
              }
              v14 = v3[2];
              v12 = v3 + 2;
              v15 = v13;
              do
              {
LABEL_14:
                if ( *v13 == v14 )
                {
                  v16 = v3[3];
                  v12 = v3 + 3;
                  goto LABEL_17;
                }
                ++v13;
              }
              while ( v9 != v13 );
            }
            goto LABEL_28;
          }
          if ( !sub_C8CA60(v2, v3[1]) )
            goto LABEL_28;
          v14 = v3[2];
          v12 = v3 + 2;
          if ( *(_BYTE *)(a1 + 84) )
          {
            v13 = *(_QWORD **)(a1 + 64);
            v9 = &v13[*(unsigned int *)(a1 + 76)];
            if ( v9 != v13 )
            {
              v15 = *(_QWORD **)(a1 + 64);
              goto LABEL_14;
            }
LABEL_28:
            v3 = v12;
            goto LABEL_29;
          }
          if ( !sub_C8CA60(v2, v3[2]) )
            goto LABEL_28;
          v16 = v3[3];
          v12 = v3 + 3;
          if ( *(_BYTE *)(a1 + 84) )
          {
            v15 = *(_QWORD **)(a1 + 64);
            v9 = &v15[*(unsigned int *)(a1 + 76)];
            if ( v9 == v15 )
              goto LABEL_28;
LABEL_17:
            while ( v16 != *v15 )
            {
              if ( v9 == ++v15 )
                goto LABEL_28;
            }
            v3 += 4;
            if ( v7 == v3 )
              goto LABEL_19;
          }
          else
          {
            if ( !sub_C8CA60(v2, v3[3]) )
              goto LABEL_28;
            v3 += 4;
            if ( v7 == v3 )
            {
LABEL_19:
              v5 = v31 - v3;
              goto LABEL_20;
            }
          }
        }
        v8 = *(_QWORD **)(a1 + 64);
        v9 = &v8[*(unsigned int *)(a1 + 76)];
        if ( v8 != v9 )
        {
          v10 = *(_QWORD **)(a1 + 64);
          do
          {
            if ( *v3 == *v10 )
            {
              v11 = v3[1];
              v12 = v3 + 1;
              v13 = *(_QWORD **)(a1 + 64);
              goto LABEL_11;
            }
            ++v10;
          }
          while ( v9 != v10 );
        }
        goto LABEL_29;
      }
LABEL_20:
      switch ( v5 )
      {
        case 2LL:
          v18 = *(_BYTE *)(a1 + 84);
          v19 = *v3;
          if ( v18 )
            goto LABEL_45;
          break;
        case 3LL:
          v18 = *(_BYTE *)(a1 + 84);
          v25 = *v3;
          if ( v18 )
          {
            v26 = *(_QWORD **)(a1 + 64);
            v27 = &v26[*(unsigned int *)(a1 + 76)];
            if ( v26 == v27 )
              goto LABEL_29;
            while ( v25 != *v26 )
            {
              if ( v27 == ++v26 )
                goto LABEL_29;
            }
            ++v3;
          }
          else
          {
            if ( !sub_C8CA60(v2, v25) )
              goto LABEL_29;
            v18 = *(_BYTE *)(a1 + 84);
            ++v3;
          }
          v19 = *v3;
          if ( v18 )
          {
LABEL_45:
            v20 = *(_QWORD **)(a1 + 64);
            v21 = &v20[*(unsigned int *)(a1 + 76)];
            if ( v20 != v21 )
            {
              while ( v19 != *v20 )
              {
                if ( v21 == ++v20 )
                  goto LABEL_29;
              }
              goto LABEL_49;
            }
            goto LABEL_29;
          }
          break;
        case 1LL:
          v22 = *v3;
          if ( !*(_BYTE *)(a1 + 84) )
          {
LABEL_56:
            if ( sub_C8CA60(v2, v22) )
              goto LABEL_23;
            goto LABEL_29;
          }
          goto LABEL_50;
        default:
          goto LABEL_23;
      }
      if ( sub_C8CA60(v2, v19) )
      {
        v18 = *(_BYTE *)(a1 + 84);
LABEL_49:
        v22 = *++v3;
        if ( !v18 )
          goto LABEL_56;
LABEL_50:
        v23 = *(_QWORD **)(a1 + 64);
        v24 = &v23[*(unsigned int *)(a1 + 76)];
        if ( v23 != v24 )
        {
          while ( v22 != *v23 )
          {
            if ( v24 == ++v23 )
              goto LABEL_29;
          }
          goto LABEL_23;
        }
      }
LABEL_29:
      if ( v31 != v3 )
      {
        if ( v30 )
          return 0;
        v30 = v29;
      }
LABEL_23:
      if ( v28 == ++v32 )
        return v30;
    }
  }
  return 0;
}

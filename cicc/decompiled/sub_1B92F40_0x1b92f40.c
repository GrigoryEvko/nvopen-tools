// Function: sub_1B92F40
// Address: 0x1b92f40
//
unsigned __int64 __fastcall sub_1B92F40(__int64 a1)
{
  __int64 v2; // rsi
  unsigned __int64 result; // rax
  __int64 v4; // r12
  __int64 *v5; // r14
  __int64 *v6; // rbx
  __int64 *v7; // r10
  __int64 *v8; // r9
  __int64 v9; // rsi
  __int64 *v10; // rdi
  unsigned int v11; // r11d
  __int64 *v12; // rcx
  _QWORD *v13; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r14
  __int64 *v16; // rax
  __int64 *i; // r13
  __int64 v18; // rsi
  __int64 *v19; // r12
  __int64 *v20; // r9
  __int64 *v21; // r8
  __int64 *v22; // rax
  __int64 *v23; // r10
  unsigned int v24; // r11d
  __int64 *v25; // rax
  __int64 *v26; // rdi
  unsigned __int64 v27; // [rsp+8h] [rbp-38h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  sub_14D04F0(*(_QWORD *)(a1 + 296), *(_QWORD *)(a1 + 352), a1 + 392);
  v2 = *(_QWORD *)(a1 + 320);
  if ( *(_DWORD *)(v2 + 88) )
  {
    v13 = *(_QWORD **)(v2 + 80);
    v14 = &v13[22 * *(unsigned int *)(v2 + 96)];
    if ( v13 != v14 )
    {
      while ( 1 )
      {
        v15 = v13;
        if ( *v13 != -8 && *v13 != -16 )
          break;
        v13 += 22;
        if ( v14 == v13 )
          goto LABEL_2;
      }
      if ( v14 != v13 )
      {
        v28 = a1 + 560;
        v16 = (__int64 *)v13[11];
        if ( v16 == (__int64 *)v15[10] )
          goto LABEL_38;
LABEL_27:
        for ( i = &v16[*((unsigned int *)v15 + 24)]; ; i = &v16[*((unsigned int *)v15 + 25)] )
        {
          if ( v16 != i )
          {
            while ( 1 )
            {
              v18 = *v16;
              v19 = v16;
              if ( (unsigned __int64)*v16 < 0xFFFFFFFFFFFFFFFELL )
                break;
              if ( i == ++v16 )
                goto LABEL_31;
            }
            if ( v16 != i )
            {
              v20 = *(__int64 **)(a1 + 576);
              v21 = *(__int64 **)(a1 + 568);
              if ( v21 != v20 )
              {
LABEL_41:
                sub_16CCBA0(v28, v18);
                v20 = *(__int64 **)(a1 + 576);
                v21 = *(__int64 **)(a1 + 568);
                goto LABEL_42;
              }
              while ( 1 )
              {
                v23 = &v21[*(unsigned int *)(a1 + 588)];
                v24 = *(_DWORD *)(a1 + 588);
                if ( v23 == v21 )
                {
LABEL_56:
                  if ( v24 >= *(_DWORD *)(a1 + 584) )
                    goto LABEL_41;
                  *(_DWORD *)(a1 + 588) = v24 + 1;
                  *v23 = v18;
                  v21 = *(__int64 **)(a1 + 568);
                  ++*(_QWORD *)(a1 + 560);
                  v20 = *(__int64 **)(a1 + 576);
                }
                else
                {
                  v25 = v21;
                  v26 = 0;
                  while ( *v25 != v18 )
                  {
                    if ( *v25 == -2 )
                      v26 = v25;
                    if ( v23 == ++v25 )
                    {
                      if ( !v26 )
                        goto LABEL_56;
                      *v26 = v18;
                      v20 = *(__int64 **)(a1 + 576);
                      --*(_DWORD *)(a1 + 592);
                      v21 = *(__int64 **)(a1 + 568);
                      ++*(_QWORD *)(a1 + 560);
                      break;
                    }
                  }
                }
LABEL_42:
                v22 = v19 + 1;
                if ( v19 + 1 == i )
                  goto LABEL_31;
                v18 = *v22;
                ++v19;
                if ( (unsigned __int64)*v22 >= 0xFFFFFFFFFFFFFFFELL )
                  break;
LABEL_46:
                if ( v19 == i )
                  goto LABEL_31;
                if ( v21 != v20 )
                  goto LABEL_41;
              }
              while ( i != ++v22 )
              {
                v18 = *v22;
                v19 = v22;
                if ( (unsigned __int64)*v22 < 0xFFFFFFFFFFFFFFFELL )
                  goto LABEL_46;
              }
            }
          }
LABEL_31:
          v15 += 22;
          if ( v15 == v14 )
            break;
          while ( *v15 == -8 || *v15 == -16 )
          {
            v15 += 22;
            if ( v14 == v15 )
              goto LABEL_35;
          }
          if ( v14 == v15 )
            break;
          v16 = (__int64 *)v15[11];
          if ( v16 != (__int64 *)v15[10] )
            goto LABEL_27;
LABEL_38:
          ;
        }
LABEL_35:
        v2 = *(_QWORD *)(a1 + 320);
      }
    }
  }
LABEL_2:
  result = *(_QWORD *)(v2 + 144);
  v4 = *(_QWORD *)(v2 + 136);
  v27 = result;
  if ( result != v4 )
  {
    while ( 1 )
    {
      v5 = *(__int64 **)(v4 + 56);
      result = *(unsigned int *)(v4 + 64);
      v6 = &v5[result];
      if ( v6 != v5 )
        break;
LABEL_16:
      v4 += 88;
      if ( v27 == v4 )
        return result;
    }
    v7 = *(__int64 **)(a1 + 576);
    v8 = *(__int64 **)(a1 + 568);
    while ( 1 )
    {
LABEL_7:
      v9 = *v5;
      if ( v8 != v7 )
        goto LABEL_5;
      result = *(unsigned int *)(a1 + 588);
      v10 = &v8[result];
      v11 = *(_DWORD *)(a1 + 588);
      if ( v10 != v8 )
      {
        result = (unsigned __int64)v8;
        v12 = 0;
        while ( v9 != *(_QWORD *)result )
        {
          if ( *(_QWORD *)result == -2 )
            v12 = (__int64 *)result;
          result += 8LL;
          if ( v10 == (__int64 *)result )
          {
            if ( !v12 )
              goto LABEL_18;
            ++v5;
            *v12 = v9;
            v7 = *(__int64 **)(a1 + 576);
            --*(_DWORD *)(a1 + 592);
            v8 = *(__int64 **)(a1 + 568);
            ++*(_QWORD *)(a1 + 560);
            if ( v6 != v5 )
              goto LABEL_7;
            goto LABEL_16;
          }
        }
        goto LABEL_6;
      }
LABEL_18:
      if ( v11 < *(_DWORD *)(a1 + 584) )
      {
        *(_DWORD *)(a1 + 588) = v11 + 1;
        *v10 = v9;
        v8 = *(__int64 **)(a1 + 568);
        ++*(_QWORD *)(a1 + 560);
        v7 = *(__int64 **)(a1 + 576);
      }
      else
      {
LABEL_5:
        result = (unsigned __int64)sub_16CCBA0(a1 + 560, v9);
        v7 = *(__int64 **)(a1 + 576);
        v8 = *(__int64 **)(a1 + 568);
      }
LABEL_6:
      if ( v6 == ++v5 )
        goto LABEL_16;
    }
  }
  return result;
}

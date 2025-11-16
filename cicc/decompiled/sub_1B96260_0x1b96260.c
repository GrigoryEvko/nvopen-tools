// Function: sub_1B96260
// Address: 0x1b96260
//
__int64 *__fastcall sub_1B96260(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  unsigned __int64 v4; // rax
  __int64 *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rbx
  __int64 *result; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  char v13; // di
  unsigned int v14; // esi
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // r12
  _QWORD *v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // r13
  _QWORD *v26; // rdx
  __int64 *v27; // r15
  __int64 *v28; // rbx
  __int64 *v29; // r10
  __int64 *v30; // r9
  __int64 v31; // rsi
  __int64 *v32; // rdi
  unsigned int v33; // r8d
  __int64 *v34; // rax
  __int64 *v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rsi
  _QWORD *v38; // rdx
  __int64 *v39; // [rsp+0h] [rbp-80h]
  __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+10h] [rbp-70h]
  __int64 *v42; // [rsp+18h] [rbp-68h]
  char v43[96]; // [rsp+20h] [rbp-60h] BYREF

  v2 = a2;
  v3 = sub_13FCB50(*a1);
  v4 = sub_157EBA0(v3);
  if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
    v5 = *(__int64 **)(v4 - 8);
  else
    v5 = (__int64 *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
  v6 = *v5;
  if ( *(_BYTE *)(*v5 + 16) > 0x17u )
  {
    v7 = *(_QWORD *)(v6 + 8);
    if ( v7 )
    {
      if ( !*(_QWORD *)(v7 + 8) )
        sub_165A590((__int64)v43, a2, v6);
    }
  }
  v8 = a1[4];
  v9 = *(__int64 **)(v8 + 144);
  result = *(__int64 **)(v8 + 136);
  v39 = v9;
  v42 = result;
  if ( v9 != result )
  {
    while ( 1 )
    {
      v11 = *v42;
      v12 = 0x17FFFFFFE8LL;
      v13 = *(_BYTE *)(*v42 + 23) & 0x40;
      v14 = *(_DWORD *)(*v42 + 20) & 0xFFFFFFF;
      if ( v14 )
      {
        v15 = 24LL * *(unsigned int *)(v11 + 56) + 8;
        v16 = 0;
        do
        {
          v17 = v11 - 24LL * v14;
          if ( v13 )
            v17 = *(_QWORD *)(v11 - 8);
          if ( v3 == *(_QWORD *)(v17 + v15) )
          {
            v12 = 24 * v16;
            goto LABEL_15;
          }
          ++v16;
          v15 += 8;
        }
        while ( v14 != (_DWORD)v16 );
        v12 = 0x17FFFFFFE8LL;
      }
LABEL_15:
      if ( v13 )
        v18 = *(_QWORD *)(v11 - 8);
      else
        v18 = v11 - 24LL * v14;
      v19 = *(_QWORD *)(v18 + v12);
      v20 = *(_QWORD *)(v19 + 8);
      v40 = v19;
      if ( v20 )
      {
        v41 = v3;
        v21 = v2;
        do
        {
          v25 = sub_1648700(v20);
          if ( (_QWORD *)v11 != v25 )
          {
            v26 = *(_QWORD **)(v21 + 16);
            v23 = *(_QWORD **)(v21 + 8);
            if ( v26 == v23 )
            {
              v22 = &v23[*(unsigned int *)(v21 + 28)];
              if ( v23 == v22 )
              {
                v38 = *(_QWORD **)(v21 + 8);
              }
              else
              {
                do
                {
                  if ( v25 == (_QWORD *)*v23 )
                    break;
                  ++v23;
                }
                while ( v22 != v23 );
                v38 = v22;
              }
LABEL_33:
              while ( v38 != v23 )
              {
                if ( *v23 < 0xFFFFFFFFFFFFFFFELL )
                  goto LABEL_22;
                ++v23;
              }
              if ( v22 == v23 )
                goto LABEL_35;
            }
            else
            {
              v22 = &v26[*(unsigned int *)(v21 + 24)];
              v23 = sub_16CC9F0(v21, (__int64)v25);
              if ( v25 == (_QWORD *)*v23 )
              {
                v36 = *(_QWORD *)(v21 + 16);
                if ( v36 == *(_QWORD *)(v21 + 8) )
                  v37 = *(unsigned int *)(v21 + 28);
                else
                  v37 = *(unsigned int *)(v21 + 24);
                v38 = (_QWORD *)(v36 + 8 * v37);
                goto LABEL_33;
              }
              v24 = *(_QWORD *)(v21 + 16);
              if ( v24 == *(_QWORD *)(v21 + 8) )
              {
                v23 = (_QWORD *)(v24 + 8LL * *(unsigned int *)(v21 + 28));
                v38 = v23;
                goto LABEL_33;
              }
              v23 = (_QWORD *)(v24 + 8LL * *(unsigned int *)(v21 + 24));
LABEL_22:
              if ( v22 == v23 )
              {
LABEL_35:
                v2 = v21;
                v3 = v41;
                goto LABEL_36;
              }
            }
          }
          v20 = *(_QWORD *)(v20 + 8);
        }
        while ( v20 );
        v2 = v21;
        v3 = v41;
      }
      sub_1412190(v2, v40);
LABEL_36:
      v27 = (__int64 *)v42[7];
      v28 = &v27[*((unsigned int *)v42 + 16)];
      if ( v28 != v27 )
        break;
LABEL_49:
      v42 += 11;
      result = v42;
      if ( v39 == v42 )
        return result;
    }
    v29 = *(__int64 **)(v2 + 16);
    v30 = *(__int64 **)(v2 + 8);
    while ( 1 )
    {
LABEL_40:
      v31 = *v27;
      if ( v30 != v29 )
        goto LABEL_38;
      v32 = &v30[*(unsigned int *)(v2 + 28)];
      v33 = *(_DWORD *)(v2 + 28);
      if ( v32 != v30 )
      {
        v34 = v30;
        v35 = 0;
        while ( v31 != *v34 )
        {
          if ( *v34 == -2 )
            v35 = v34;
          if ( v32 == ++v34 )
          {
            if ( !v35 )
              goto LABEL_55;
            ++v27;
            *v35 = v31;
            v29 = *(__int64 **)(v2 + 16);
            --*(_DWORD *)(v2 + 32);
            v30 = *(__int64 **)(v2 + 8);
            ++*(_QWORD *)v2;
            if ( v28 != v27 )
              goto LABEL_40;
            goto LABEL_49;
          }
        }
        goto LABEL_39;
      }
LABEL_55:
      if ( v33 < *(_DWORD *)(v2 + 24) )
      {
        *(_DWORD *)(v2 + 28) = v33 + 1;
        *v32 = v31;
        v30 = *(__int64 **)(v2 + 8);
        ++*(_QWORD *)v2;
        v29 = *(__int64 **)(v2 + 16);
      }
      else
      {
LABEL_38:
        sub_16CCBA0(v2, v31);
        v29 = *(__int64 **)(v2 + 16);
        v30 = *(__int64 **)(v2 + 8);
      }
LABEL_39:
      if ( v28 == ++v27 )
        goto LABEL_49;
    }
  }
  return result;
}

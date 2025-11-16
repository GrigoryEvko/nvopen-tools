// Function: sub_148FD90
// Address: 0x148fd90
//
char __fastcall sub_148FD90(__int64 *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // r12
  char result; // al
  __int64 *v12; // r8
  __int64 *v13; // rbx
  __int64 *i; // r15
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 *v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  char *v20; // rdi
  __int64 *v21; // rax
  __int64 *v22; // rsi
  size_t v23; // rdx
  __int64 *v24; // rbx
  __int64 v25; // rax
  __int64 *v28; // [rsp+8h] [rbp-68h]
  __int64 *v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h] BYREF
  __int64 v31; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v32; // [rsp+20h] [rbp-50h] BYREF
  __int64 v33; // [rsp+28h] [rbp-48h]
  _BYTE v34[64]; // [rsp+30h] [rbp-40h] BYREF

  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(__int64 **)a2;
  v8 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * ((int)v6 - 1));
  v30 = v8;
  if ( (_DWORD)v6 == 1 )
  {
    if ( *(_WORD *)(v8 + 24) == 5 )
    {
      v12 = (__int64 *)v34;
      v32 = (__int64 *)v34;
      v33 = 0x200000000LL;
      v13 = *(__int64 **)(v8 + 32);
      for ( i = &v13[*(_QWORD *)(v8 + 40)]; i != v13; LODWORD(v33) = v33 + 1 )
      {
        while ( 1 )
        {
          v15 = *v13;
          if ( *(_WORD *)(*v13 + 24) )
            break;
          if ( i == ++v13 )
            goto LABEL_16;
        }
        v16 = (unsigned int)v33;
        if ( (unsigned int)v33 >= HIDWORD(v33) )
        {
          v29 = v12;
          sub_16CD150(&v32, v12, 0, 8);
          v16 = (unsigned int)v33;
          v12 = v29;
        }
        ++v13;
        v32[v16] = v15;
      }
LABEL_16:
      v28 = v12;
      v30 = sub_147EE30(a1, &v32, 0, 0, a4, a5);
      if ( v32 != v28 )
        _libc_free((unsigned __int64)v32);
    }
    goto LABEL_18;
  }
  v9 = v6;
  v10 = &v7[v9];
  if ( v7 == &v7[v9] )
  {
    v17 = &v7[v9];
LABEL_20:
    v18 = (v9 * 8) >> 3;
    v19 = (v9 * 8) >> 5;
    if ( v19 )
    {
      v20 = (char *)v10;
      v21 = &v10[4 * v19];
      while ( *(_WORD *)(*(_QWORD *)v20 + 24LL) )
      {
        if ( !*(_WORD *)(*((_QWORD *)v20 + 1) + 24LL) )
        {
          v20 += 8;
          goto LABEL_27;
        }
        if ( !*(_WORD *)(*((_QWORD *)v20 + 2) + 24LL) )
        {
          v20 += 16;
          goto LABEL_27;
        }
        if ( !*(_WORD *)(*((_QWORD *)v20 + 3) + 24LL) )
        {
          v20 += 24;
          goto LABEL_27;
        }
        v20 += 32;
        if ( v21 == (__int64 *)v20 )
        {
          v18 = ((char *)v17 - v20) >> 3;
          goto LABEL_41;
        }
      }
      goto LABEL_27;
    }
    v20 = (char *)v10;
LABEL_41:
    if ( v18 != 2 )
    {
      if ( v18 != 3 )
      {
        v24 = v17;
        if ( v18 != 1 )
          goto LABEL_34;
        goto LABEL_44;
      }
      if ( !*(_WORD *)(*(_QWORD *)v20 + 24LL) )
      {
LABEL_27:
        if ( v20 == (char *)v17 || (v22 = (__int64 *)(v20 + 8), v20 + 8 == (char *)v17) )
        {
          v24 = (__int64 *)v20;
        }
        else
        {
          do
          {
            if ( *(_WORD *)(*v22 + 24) )
            {
              *(_QWORD *)v20 = *v22;
              v20 += 8;
            }
            ++v22;
          }
          while ( v22 != v17 );
          v10 = *(__int64 **)a2;
          v23 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - (_QWORD)v22;
          v24 = (__int64 *)&v20[v23];
          if ( (__int64 *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8)) != v22 )
          {
            memmove(v20, v22, v23);
            v10 = *(__int64 **)a2;
          }
        }
LABEL_34:
        v25 = v24 - v10;
        *(_DWORD *)(a2 + 8) = v25;
        if ( (_DWORD)v25 )
        {
          result = sub_148FD90(a1, a2, a3);
          if ( !result )
            return result;
        }
LABEL_18:
        sub_1458920(a3, &v30);
        return 1;
      }
      v20 += 8;
    }
    if ( *(_WORD *)(*(_QWORD *)v20 + 24LL) )
    {
      v20 += 8;
LABEL_44:
      v24 = v17;
      if ( *(_WORD *)(*(_QWORD *)v20 + 24LL) )
        goto LABEL_34;
      goto LABEL_27;
    }
    goto LABEL_27;
  }
  while ( 1 )
  {
    sub_148F0C0(a1, *v7, v8, &v31, (__int64 *)&v32, a4, a5);
    result = sub_14560B0((__int64)v32);
    if ( !result )
      return result;
    *v7++ = v31;
    if ( v10 == v7 )
    {
      v10 = *(__int64 **)a2;
      v9 = *(unsigned int *)(a2 + 8);
      v17 = (__int64 *)(*(_QWORD *)a2 + v9 * 8);
      goto LABEL_20;
    }
    v8 = v30;
  }
}

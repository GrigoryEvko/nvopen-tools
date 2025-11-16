// Function: sub_1F33C80
// Address: 0x1f33c80
//
__int64 *__fastcall sub_1F33C80(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 *result; // rax
  __int64 v7; // r13
  __int64 v8; // rcx
  int v9; // esi
  __int64 v10; // rdx
  unsigned int i; // ebx
  unsigned int v12; // r15d
  int v13; // r14d
  __int64 v14; // rcx
  __int64 v15; // rax
  int v16; // r11d
  unsigned int v17; // esi
  __int64 v18; // r12
  int v19; // edi
  __int64 v20; // rax
  __int64 j; // r14
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  _BYTE *v27; // r9
  __int64 v28; // rbx
  __int64 v29; // rdi
  unsigned int v30; // ebx
  __int64 v31; // r15
  __int64 v32; // r14
  __int64 v33; // r12
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  _BYTE *v38; // r9
  __int64 *v40; // [rsp+10h] [rbp-B0h]
  __int64 v43; // [rsp+28h] [rbp-98h]
  __int64 *v44; // [rsp+30h] [rbp-90h]
  int v46; // [rsp+3Ch] [rbp-84h]
  __int64 v47; // [rsp+40h] [rbp-80h]
  __int64 v48; // [rsp+48h] [rbp-78h]
  __int64 v49; // [rsp+58h] [rbp-68h]
  __m128i v50; // [rsp+60h] [rbp-60h] BYREF
  __int64 v51; // [rsp+70h] [rbp-50h]
  __int64 v52; // [rsp+78h] [rbp-48h]
  __int64 v53; // [rsp+80h] [rbp-40h]

  result = *(__int64 **)(a5 + 80);
  v40 = &result[*(unsigned int *)(a5 + 88)];
  v44 = result;
  if ( result != v40 )
  {
    while ( 1 )
    {
      v48 = *v44;
      v43 = *v44 + 24;
      if ( *(_QWORD *)(*v44 + 32) != v43 )
        break;
LABEL_4:
      result = ++v44;
      if ( v40 == v44 )
        return result;
    }
    v7 = *(_QWORD *)(*v44 + 32);
    while ( 1 )
    {
      if ( **(_WORD **)(v7 + 16) != 45 && **(_WORD **)(v7 + 16) )
        goto LABEL_4;
      v8 = a2;
      v9 = *(_DWORD *)(v7 + 40);
      v10 = *(_QWORD *)(v7 + 32);
      v47 = *(_QWORD *)(a2 + 56);
      if ( v9 == 1 )
        break;
      for ( i = 1; i != v9; i += 2 )
      {
        if ( *(_QWORD *)(v10 + 40LL * (i + 1) + 24) == a2 )
        {
          v12 = 0;
          v13 = *(_DWORD *)(v10 + 40LL * i + 8);
          if ( !a3 )
            goto LABEL_13;
          goto LABEL_42;
        }
      }
      i = 0;
      v12 = 0;
      v13 = *(_DWORD *)(v10 + 8);
      if ( !a3 )
        goto LABEL_13;
LABEL_42:
      v12 = v9 - 2;
      v33 = a2;
      if ( i != v9 - 2 )
        goto LABEL_46;
LABEL_13:
      v14 = a1;
      v15 = *(unsigned int *)(a1 + 160);
      if ( (_DWORD)v15 )
      {
        a6 = (unsigned int)(v15 - 1);
        v10 = *(_QWORD *)(a1 + 144);
        v16 = 1;
        v17 = a6 & (37 * v13);
        v18 = v10 + 32LL * v17;
        v19 = *(_DWORD *)v18;
        if ( v13 == *(_DWORD *)v18 )
        {
LABEL_15:
          if ( v18 != v10 + 32 * v15 )
          {
            v20 = *(_QWORD *)(v18 + 8);
            v10 = (*(_QWORD *)(v18 + 16) - v20) >> 4;
            if ( (_DWORD)v10 )
            {
              v49 = 16LL * (unsigned int)(v10 - 1);
              for ( j = 0; ; j += 16 )
              {
                v23 = *(_QWORD *)(v20 + j);
                if ( !sub_1DD6970(v23, v48) )
                  goto LABEL_19;
                if ( v12 )
                  break;
                v50.m128i_i32[2] = *(_DWORD *)(*(_QWORD *)(v18 + 8) + j + 8);
                v50.m128i_i64[0] = 0;
                v51 = 0;
                v52 = 0;
                v53 = 0;
                sub_1E1A9C0(v7, v47, &v50);
                v50.m128i_i8[0] = 4;
                v51 = 0;
                v50.m128i_i32[0] &= 0xFFF000FF;
                v52 = v23;
                sub_1E1A9C0(v7, v47, &v50);
                if ( v49 == j )
                  goto LABEL_24;
LABEL_20:
                v20 = *(_QWORD *)(v18 + 8);
              }
              sub_1E310D0(*(_QWORD *)(v7 + 32) + 40LL * v12, *(_DWORD *)(*(_QWORD *)(v18 + 8) + j + 8));
              v10 = *(_QWORD *)(v7 + 32);
              v22 = v12 + 1;
              v12 = 0;
              *(_QWORD *)(v10 + 40 * v22 + 24) = v23;
LABEL_19:
              if ( v49 == j )
                goto LABEL_24;
              goto LABEL_20;
            }
LABEL_24:
            if ( v12 )
            {
              sub_1E16C90(v7, v12 + 1, v10, v14, a5, (_BYTE *)a6);
              sub_1E16C90(v7, v12, v24, v25, v26, v27);
            }
            goto LABEL_26;
          }
        }
        else
        {
          while ( v19 != -1 )
          {
            v14 = (unsigned int)(v16 + 1);
            v17 = a6 & (v16 + v17);
            v18 = v10 + 32LL * v17;
            v19 = *(_DWORD *)v18;
            if ( *(_DWORD *)v18 == v13 )
              goto LABEL_15;
            ++v16;
          }
        }
      }
      v28 = *(unsigned int *)(a4 + 8);
      if ( !(_DWORD)v28 )
        goto LABEL_24;
      v29 = 8 * v28;
      v30 = v12;
      v46 = v13;
      v31 = 0;
      do
      {
        v32 = *(_QWORD *)(*(_QWORD *)a4 + v31);
        if ( v30 )
        {
          sub_1E310D0(*(_QWORD *)(v7 + 32) + 40LL * v30, v46);
          *(_QWORD *)(*(_QWORD *)(v7 + 32) + 40LL * (v30 + 1) + 24) = v32;
        }
        else
        {
          v50.m128i_i64[0] = 0;
          v51 = 0;
          v50.m128i_i32[2] = v46;
          v52 = 0;
          v53 = 0;
          sub_1E1A9C0(v7, v47, &v50);
          v50.m128i_i8[0] = 4;
          v51 = 0;
          v50.m128i_i32[0] &= 0xFFF000FF;
          v52 = v32;
          sub_1E1A9C0(v7, v47, &v50);
        }
        v31 += 8;
        v30 = 0;
      }
      while ( v29 != v31 );
LABEL_26:
      if ( (*(_BYTE *)v7 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v7 + 46) & 8) != 0 )
          v7 = *(_QWORD *)(v7 + 8);
      }
      v7 = *(_QWORD *)(v7 + 8);
      if ( v43 == v7 )
        goto LABEL_4;
    }
    v12 = 0;
    v13 = *(_DWORD *)(v10 + 8);
    if ( !a3 )
      goto LABEL_13;
    v33 = a2;
    i = 0;
    v12 = -1;
    while ( 1 )
    {
LABEL_46:
      v34 = v12 + 1;
      if ( v33 == *(_QWORD *)(v10 + 40 * v34 + 24) )
      {
        sub_1E16C90(v7, v34, v10, v8, a5, (_BYTE *)a6);
        sub_1E16C90(v7, v12, v35, v36, v37, v38);
      }
      v12 -= 2;
      if ( i == v12 )
        break;
      v10 = *(_QWORD *)(v7 + 32);
    }
    goto LABEL_13;
  }
  return result;
}

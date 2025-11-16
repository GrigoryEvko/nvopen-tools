// Function: sub_2E7FFC0
// Address: 0x2e7ffc0
//
void __fastcall sub_2E7FFC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r8
  __int64 *v7; // rbx
  __int64 *v8; // r14
  unsigned int v9; // eax
  __int64 v10; // rdi
  __int64 *v11; // rbx
  __int64 *v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // rdi
  unsigned int v15; // eax
  int v16; // r9d
  int v17; // r11d
  __int64 *v18; // r9
  int v19; // eax
  __int64 v20; // r15
  int v21; // esi
  int v22; // esi
  __int64 v23; // r15
  __int64 v24; // [rsp+0h] [rbp-50h] BYREF
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+18h] [rbp-38h]

  v4 = 0;
  v5 = 0;
  v7 = *(__int64 **)(a1 + 8);
  v8 = *(__int64 **)(a1 + 16);
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      while ( !*((_BYTE *)v7 + 9) )
      {
        v7 += 2;
        if ( v8 == v7 )
          goto LABEL_10;
      }
      if ( !(_DWORD)v4 )
        break;
      a3 = *v7;
      v9 = (v4 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
      a4 = v5 + 8LL * v9;
      v10 = *(_QWORD *)a4;
      if ( *v7 != *(_QWORD *)a4 )
      {
        v17 = 1;
        v18 = 0;
        while ( v10 != -4096 )
        {
          if ( v18 || v10 != -8192 )
            a4 = (__int64)v18;
          v9 = (v4 - 1) & (v17 + v9);
          v10 = *(_QWORD *)(v5 + 8LL * v9);
          if ( a3 == v10 )
            goto LABEL_7;
          ++v17;
          v18 = (__int64 *)a4;
          a4 = v5 + 8LL * v9;
        }
        if ( !v18 )
          v18 = (__int64 *)a4;
        ++v24;
        v19 = v26 + 1;
        if ( 4 * ((int)v26 + 1) < (unsigned int)(3 * v4) )
        {
          a4 = (unsigned int)(v4 - (v19 + HIDWORD(v26)));
          a3 = (unsigned int)v4 >> 3;
          if ( (unsigned int)a4 <= (unsigned int)a3 )
          {
            sub_2E7FB10((__int64)&v24, v4);
            if ( !(_DWORD)v27 )
            {
LABEL_68:
              LODWORD(v26) = v26 + 1;
              BUG();
            }
            v22 = 1;
            a4 = 0;
            a3 = ((_DWORD)v27 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
            v18 = (__int64 *)(v25 + 8 * a3);
            v23 = *v18;
            v19 = v26 + 1;
            if ( *v7 != *v18 )
            {
              while ( v23 != -4096 )
              {
                if ( !a4 && v23 == -8192 )
                  a4 = (__int64)v18;
                a3 = ((_DWORD)v27 - 1) & (unsigned int)(v22 + a3);
                v18 = (__int64 *)(v25 + 8LL * (unsigned int)a3);
                v23 = *v18;
                if ( *v7 == *v18 )
                  goto LABEL_39;
                ++v22;
              }
              goto LABEL_55;
            }
          }
          goto LABEL_39;
        }
LABEL_43:
        sub_2E7FB10((__int64)&v24, 2 * v4);
        if ( !(_DWORD)v27 )
          goto LABEL_68;
        a3 = ((_DWORD)v27 - 1) & (((unsigned int)*v7 >> 9) ^ ((unsigned int)*v7 >> 4));
        v18 = (__int64 *)(v25 + 8 * a3);
        v20 = *v18;
        v19 = v26 + 1;
        if ( *v7 != *v18 )
        {
          v21 = 1;
          a4 = 0;
          while ( v20 != -4096 )
          {
            if ( v20 == -8192 && !a4 )
              a4 = (__int64)v18;
            a3 = ((_DWORD)v27 - 1) & (unsigned int)(v21 + a3);
            v18 = (__int64 *)(v25 + 8LL * (unsigned int)a3);
            v20 = *v18;
            if ( *v7 == *v18 )
              goto LABEL_39;
            ++v21;
          }
LABEL_55:
          if ( a4 )
            v18 = (__int64 *)a4;
        }
LABEL_39:
        LODWORD(v26) = v19;
        if ( *v18 != -4096 )
          --HIDWORD(v26);
        v10 = *v7;
        *v18 = *v7;
      }
LABEL_7:
      if ( v10 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 16LL))(v10);
      v7 += 2;
      v5 = v25;
      v4 = (unsigned int)v27;
      if ( v8 == v7 )
        goto LABEL_10;
    }
    ++v24;
    goto LABEL_43;
  }
LABEL_10:
  v11 = *(__int64 **)(a1 + 40);
  v12 = &v11[*(unsigned int *)(a1 + 56)];
  if ( *(_DWORD *)(a1 + 48) && v11 != v12 )
  {
    while ( *v11 == -8192 || *v11 == -4096 )
    {
      if ( ++v11 == v12 )
        goto LABEL_11;
    }
    if ( v12 != v11 )
    {
      v14 = *v11;
      if ( !(_DWORD)v4 )
        goto LABEL_27;
      while ( 1 )
      {
        a4 = (unsigned int)(v4 - 1);
        v15 = a4 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        a3 = *(_QWORD *)(v5 + 8LL * v15);
        if ( v14 != a3 )
        {
          v16 = 1;
          while ( a3 != -4096 )
          {
            v15 = a4 & (v16 + v15);
            a3 = *(_QWORD *)(v5 + 8LL * v15);
            if ( v14 == a3 )
              goto LABEL_22;
            ++v16;
          }
LABEL_27:
          if ( v14 )
          {
            (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v14 + 16LL))(
              v14,
              v4,
              a3,
              a4,
              v5);
            v4 = (unsigned int)v27;
            v5 = v25;
          }
        }
        do
        {
LABEL_22:
          if ( ++v11 == v12 )
            goto LABEL_11;
        }
        while ( *v11 == -4096 || *v11 == -8192 );
        if ( v11 == v12 )
          break;
        v14 = *v11;
        if ( !(_DWORD)v4 )
          goto LABEL_27;
      }
    }
  }
LABEL_11:
  sub_C7D6A0(v5, 8 * v4, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 40), 8LL * *(unsigned int *)(a1 + 56), 8);
  v13 = *(_QWORD *)(a1 + 8);
  if ( v13 )
    j_j___libc_free_0(v13);
}

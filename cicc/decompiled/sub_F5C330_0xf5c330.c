// Function: sub_F5C330
// Address: 0xf5c330
//
__int64 __fastcall sub_F5C330(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4)
{
  __int64 result; // rax
  unsigned __int64 v6; // rcx
  _QWORD *v7; // rsi
  __int64 v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r14
  unsigned __int64 i; // r12
  unsigned __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // rsi
  unsigned __int64 *v23; // r10
  int v24; // eax
  unsigned __int64 *v25; // rdi
  unsigned __int64 v26; // rax
  int v27; // ecx
  __int64 v28; // rsi
  __int64 v29; // rcx
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rsi
  char *v34; // r13
  int v35; // eax
  unsigned __int64 v39; // [rsp+30h] [rbp-50h] BYREF
  __int64 v40; // [rsp+38h] [rbp-48h]
  unsigned __int64 v41; // [rsp+40h] [rbp-40h]

  while ( 1 )
  {
    result = *(unsigned int *)(a1 + 8);
    if ( !(_DWORD)result )
      return result;
    while ( 1 )
    {
      v6 = *(_QWORD *)a1;
      v39 = 6;
      v40 = 0;
      v7 = (_QWORD *)(v6 + 24LL * (unsigned int)result - 24);
      v41 = v7[2];
      if ( v41 != -4096 && v41 != 0 && v41 != -8192 )
      {
        sub_BD6050(&v39, *v7 & 0xFFFFFFFFFFFFFFF8LL);
        v6 = *(_QWORD *)a1;
        LODWORD(result) = *(_DWORD *)(a1 + 8);
      }
      v8 = (unsigned int)(result - 1);
      *(_DWORD *)(a1 + 8) = v8;
      v9 = (_QWORD *)(v6 + 24 * v8);
      v10 = v9[2];
      if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
        sub_BD60C0(v9);
      v11 = v41;
      if ( !v41 )
        break;
      if ( v41 != -4096 && v41 != -8192 )
        sub_BD60C0(&v39);
      sub_F54ED0((unsigned __int8 *)v11);
      if ( *(_QWORD *)(a4 + 16) )
      {
        v39 = v11;
        (*(void (__fastcall **)(__int64, unsigned __int64 *))(a4 + 24))(a4, &v39);
      }
      v14 = 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v11 + 7) & 0x40) != 0 )
      {
        v15 = *(_QWORD *)(v11 - 8);
        v16 = v15 + v14;
      }
      else
      {
        v16 = v11;
        v15 = v11 - v14;
      }
      for ( i = v15; v16 != i; i += 32LL )
      {
        while ( 1 )
        {
          v18 = *(_QWORD *)i;
          if ( *(_QWORD *)i )
          {
            v19 = *(_QWORD *)(i + 8);
            **(_QWORD **)(i + 16) = v19;
            if ( v19 )
              *(_QWORD *)(v19 + 16) = *(_QWORD *)(i + 16);
          }
          *(_QWORD *)i = 0;
          if ( !*(_QWORD *)(v18 + 16) && *(_BYTE *)v18 > 0x1Cu && sub_F50EE0((unsigned __int8 *)v18, a2) )
          {
            v39 = 6;
            v40 = 0;
            v41 = v18;
            if ( v18 != -4096 && v18 != -8192 )
              sub_BD73F0((__int64)&v39);
            v21 = *(unsigned int *)(a1 + 8);
            v22 = *(_QWORD *)a1;
            v23 = &v39;
            v24 = *(_DWORD *)(a1 + 8);
            if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
            {
              if ( v22 > (unsigned __int64)&v39 || (unsigned __int64)&v39 >= v22 + 24 * v21 )
              {
                sub_F39130(a1, v21 + 1, v20, v21, v12, v13);
                v21 = *(unsigned int *)(a1 + 8);
                v22 = *(_QWORD *)a1;
                v23 = &v39;
                v24 = *(_DWORD *)(a1 + 8);
              }
              else
              {
                v34 = (char *)&v39 - v22;
                sub_F39130(a1, v21 + 1, v20, v21, v12, v13);
                v22 = *(_QWORD *)a1;
                v21 = *(unsigned int *)(a1 + 8);
                v23 = (unsigned __int64 *)&v34[*(_QWORD *)a1];
                v24 = *(_DWORD *)(a1 + 8);
              }
            }
            v25 = (unsigned __int64 *)(v22 + 24 * v21);
            if ( v25 )
            {
              *v25 = 6;
              v26 = v23[2];
              v25[1] = 0;
              v25[2] = v26;
              if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
                sub_BD6050(v25, *v23 & 0xFFFFFFFFFFFFFFF8LL);
              v24 = *(_DWORD *)(a1 + 8);
            }
            *(_DWORD *)(a1 + 8) = v24 + 1;
            if ( v41 != -4096 && v41 != 0 && v41 != -8192 )
              break;
          }
          i += 32LL;
          if ( v16 == i )
            goto LABEL_37;
        }
        sub_BD60C0(&v39);
      }
LABEL_37:
      if ( a3 )
      {
        v27 = *(_DWORD *)(*a3 + 56LL);
        v28 = *(_QWORD *)(*a3 + 40LL);
        if ( v27 )
        {
          v29 = (unsigned int)(v27 - 1);
          v30 = v29 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v31 = (__int64 *)(v28 + 16LL * v30);
          v32 = *v31;
          if ( v11 == *v31 )
          {
LABEL_40:
            v33 = v31[1];
            if ( v33 )
              sub_D6E4B0(a3, v33, 0, v29, v12, v13);
          }
          else
          {
            v35 = 1;
            while ( v32 != -4096 )
            {
              v13 = (unsigned int)(v35 + 1);
              v30 = v29 & (v35 + v30);
              v31 = (__int64 *)(v28 + 16LL * v30);
              v32 = *v31;
              if ( v11 == *v31 )
                goto LABEL_40;
              v35 = v13;
            }
          }
        }
      }
      sub_B43D60((_QWORD *)v11);
      result = *(unsigned int *)(a1 + 8);
      if ( !(_DWORD)result )
        return result;
    }
  }
}

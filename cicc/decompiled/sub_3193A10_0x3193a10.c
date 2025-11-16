// Function: sub_3193A10
// Address: 0x3193a10
//
__int64 *__fastcall sub_3193A10(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 *result; // rax
  __int64 *v6; // r15
  __int64 *v7; // r14
  const char *v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 *v12; // rbx
  __int64 *v13; // r13
  __int64 v14; // r15
  unsigned __int8 *v15; // r10
  int v16; // esi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int v20; // r10d
  int v21; // eax
  int v22; // eax
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  int v27; // eax
  int v28; // eax
  unsigned int v29; // edx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  int v34; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v37; // [rsp+30h] [rbp-80h]
  __int64 v38; // [rsp+38h] [rbp-78h]
  char v39; // [rsp+38h] [rbp-78h]
  __int64 *v40; // [rsp+48h] [rbp-68h]
  const char *v41[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v42; // [rsp+70h] [rbp-40h]

  result = (__int64 *)(a1 + 48);
  v6 = *(__int64 **)(a1 + 56);
  v34 = a4;
  if ( v6 != (__int64 *)(a1 + 48) )
  {
    result = &a3[3 * a4];
    v40 = result;
    do
    {
      if ( !v6 )
        BUG();
      v7 = v6 - 3;
      if ( *((_BYTE *)v6 - 24) != 84 )
        return result;
      v38 = *(_QWORD *)(a5 + 56);
      v41[0] = sub_BD5D20((__int64)(v6 - 3));
      v42 = 773;
      v41[1] = v8;
      v41[2] = ".moved";
      v9 = *(v6 - 2);
      v10 = sub_BD2DA0(80);
      v11 = v10;
      if ( v10 )
      {
        sub_B44260(v10, v9, 55, 0x8000000u, v38, 1u);
        *(_DWORD *)(v11 + 72) = v34;
        sub_BD6B50((unsigned __int8 *)v11, v41);
        sub_BD2A10(v11, *(_DWORD *)(v11 + 72), 1);
      }
      v39 = 1;
      v12 = v6;
      v13 = a3;
      if ( a3 == v40 )
        goto LABEL_43;
      do
      {
        v14 = *v13;
        v15 = (unsigned __int8 *)sub_ACADE0((__int64 **)*(v12 - 2));
        if ( a1 == v14 )
        {
          v15 = (unsigned __int8 *)v11;
        }
        else
        {
          v16 = *((_DWORD *)v12 - 5) & 0x7FFFFFF;
          if ( v16 )
          {
            v17 = 0;
            v18 = *(v12 - 4) + 32LL * *((unsigned int *)v12 + 12);
            while ( v14 != *(_QWORD *)(v18 + 8 * v17) )
            {
              if ( v16 == (_DWORD)++v17 )
                goto LABEL_17;
            }
            v19 = 0;
            while ( 1 )
            {
              v20 = v19;
              if ( v14 == *(_QWORD *)(v18 + 8 * v19) )
                break;
              if ( v16 == (_DWORD)++v19 )
              {
                v20 = -1;
                break;
              }
            }
            v15 = (unsigned __int8 *)sub_B48BF0((__int64)v7, v20, 0);
            v39 &= (unsigned int)*v15 - 12 <= 1;
          }
        }
LABEL_17:
        v21 = *(_DWORD *)(v11 + 4) & 0x7FFFFFF;
        if ( v21 == *(_DWORD *)(v11 + 72) )
        {
          v37 = v15;
          sub_B48D90(v11);
          v15 = v37;
          v21 = *(_DWORD *)(v11 + 4) & 0x7FFFFFF;
        }
        v22 = (v21 + 1) & 0x7FFFFFF;
        v23 = v22 | *(_DWORD *)(v11 + 4) & 0xF8000000;
        v24 = *(_QWORD *)(v11 - 8) + 32LL * (unsigned int)(v22 - 1);
        *(_DWORD *)(v11 + 4) = v23;
        if ( *(_QWORD *)v24 )
        {
          v25 = *(_QWORD *)(v24 + 8);
          **(_QWORD **)(v24 + 16) = v25;
          if ( v25 )
            *(_QWORD *)(v25 + 16) = *(_QWORD *)(v24 + 16);
        }
        *(_QWORD *)v24 = v15;
        if ( v15 )
        {
          v26 = *((_QWORD *)v15 + 2);
          *(_QWORD *)(v24 + 8) = v26;
          if ( v26 )
            *(_QWORD *)(v26 + 16) = v24 + 8;
          *(_QWORD *)(v24 + 16) = v15 + 16;
          *((_QWORD *)v15 + 2) = v24;
        }
        v13 += 3;
        *(_QWORD *)(*(_QWORD *)(v11 - 8)
                  + 32LL * *(unsigned int *)(v11 + 72)
                  + 8LL * ((*(_DWORD *)(v11 + 4) & 0x7FFFFFFu) - 1)) = v14;
      }
      while ( v40 != v13 );
      v6 = v12;
      if ( v39 )
      {
LABEL_43:
        sub_B43D60((_QWORD *)v11);
        v11 = sub_ACADE0((__int64 **)*(v6 - 2));
        v27 = *((_DWORD *)v6 - 5) & 0x7FFFFFF;
        if ( v27 )
        {
LABEL_29:
          if ( v27 == *((_DWORD *)v6 + 12) )
          {
            sub_B48D90((__int64)v7);
            v27 = *((_DWORD *)v6 - 5) & 0x7FFFFFF;
          }
          v28 = (v27 + 1) & 0x7FFFFFF;
          v29 = v28 | *((_DWORD *)v6 - 5) & 0xF8000000;
          v30 = *(v6 - 4) + 32LL * (unsigned int)(v28 - 1);
          *((_DWORD *)v6 - 5) = v29;
          if ( *(_QWORD *)v30 )
          {
            v31 = *(_QWORD *)(v30 + 8);
            **(_QWORD **)(v30 + 16) = v31;
            if ( v31 )
              *(_QWORD *)(v31 + 16) = *(_QWORD *)(v30 + 16);
          }
          *(_QWORD *)v30 = v11;
          if ( v11 )
          {
            v32 = *(_QWORD *)(v11 + 16);
            *(_QWORD *)(v30 + 8) = v32;
            if ( v32 )
              *(_QWORD *)(v32 + 16) = v30 + 8;
            *(_QWORD *)(v30 + 16) = v11 + 16;
            *(_QWORD *)(v11 + 16) = v30;
          }
          result = (__int64 *)(*(v6 - 4)
                             + 32LL * *((unsigned int *)v6 + 12)
                             + 8LL * ((*((_DWORD *)v6 - 5) & 0x7FFFFFFu) - 1));
          *result = a2;
          v6 = (__int64 *)v6[1];
          continue;
        }
      }
      else
      {
        v27 = *((_DWORD *)v12 - 5) & 0x7FFFFFF;
        if ( v27 )
          goto LABEL_29;
      }
      sub_BD84D0((__int64)v7, v11);
      result = (__int64 *)sub_B43D60(v7);
      v6 = result;
    }
    while ( v6 != (__int64 *)(a1 + 48) );
  }
  return result;
}

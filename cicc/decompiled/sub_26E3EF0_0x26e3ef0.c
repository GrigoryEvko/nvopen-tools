// Function: sub_26E3EF0
// Address: 0x26e3ef0
//
_QWORD *__fastcall sub_26E3EF0(__int64 a1, __int64 *a2)
{
  int *v4; // r13
  size_t v5; // r12
  int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // r12
  _DWORD *v11; // rax
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rcx
  __int64 v15; // r13
  _DWORD *v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // rbx
  __int64 v20; // r8
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // r15
  int v24; // edx
  __int64 v25; // rsi
  int v26; // ecx
  unsigned int v27; // eax
  __int64 *v28; // rdx
  __int64 v29; // rdi
  int v30; // edx
  int v31; // r8d
  _QWORD *v32; // [rsp+8h] [rbp-108h]
  int v33; // [rsp+14h] [rbp-FCh]
  bool v34; // [rsp+18h] [rbp-F8h]
  __int64 v35; // [rsp+18h] [rbp-F8h]
  _QWORD *i; // [rsp+20h] [rbp-F0h]
  _QWORD *v37; // [rsp+28h] [rbp-E8h]
  size_t v38[2]; // [rsp+30h] [rbp-E0h] BYREF
  int v39[52]; // [rsp+40h] [rbp-D0h] BYREF

  v4 = (int *)a2[2];
  v5 = a2[3];
  if ( !unk_4F838D1 )
  {
    if ( !v4 )
      v5 = 0;
    goto LABEL_4;
  }
  v23 = *a2;
  if ( v4 )
  {
    sub_C7D030(v39);
    sub_C7D280(v39, v4, v5);
    sub_C7D290(v39, v38);
    v5 = v38[0];
  }
  v24 = *(_DWORD *)(v23 + 24);
  v25 = *(_QWORD *)(v23 + 8);
  if ( v24 )
  {
    v26 = v24 - 1;
    v27 = (v24 - 1) & (((0xBF58476D1CE4E5B9LL * v5) >> 31) ^ (484763065 * v5));
    v28 = (__int64 *)(v25 + 24LL * v27);
    v29 = *v28;
    if ( v5 == *v28 )
    {
LABEL_39:
      v4 = (int *)v28[1];
      v5 = v28[2];
      goto LABEL_4;
    }
    v30 = 1;
    while ( v29 != -1 )
    {
      v31 = v30 + 1;
      v27 = v26 & (v30 + v27);
      v28 = (__int64 *)(v25 + 24LL * v27);
      v29 = *v28;
      if ( v5 == *v28 )
        goto LABEL_39;
      v30 = v31;
    }
  }
  v5 = 0;
  v4 = 0;
LABEL_4:
  v6 = sub_C92610();
  result = (_QWORD *)sub_C92860((__int64 *)(a1 + 120), v4, v5, v6);
  if ( (_DWORD)result != -1 )
  {
    v8 = *(_QWORD *)(a1 + 120);
    result = (_QWORD *)(v8 + 8LL * (int)result);
    if ( result != (_QWORD *)(v8 + 8LL * *(unsigned int *)(a1 + 128)) )
    {
      result = (_QWORD *)*result;
      v37 = result;
      if ( result[4] )
      {
        v9 = a2[12];
        v10 = a2 + 10;
        for ( i = result + 1; v10 != (_QWORD *)v9; v9 = sub_220EF30(v9) )
        {
          while ( 1 )
          {
            v11 = sub_26E3C30(i, *(_QWORD *)(v9 + 32) % v37[2], (_DWORD *)(v9 + 32), *(_QWORD *)(v9 + 32));
            if ( v11 )
            {
              v12 = *(_QWORD *)v11;
              if ( v12 )
                break;
            }
LABEL_14:
            v9 = sub_220EF30(v9);
            if ( v10 == (_QWORD *)v9 )
              goto LABEL_15;
          }
          v13 = *(_DWORD *)(v12 + 16);
          v14 = *(_QWORD *)(v9 + 40);
          if ( (v13 & 0xFFFFFFFB) != 2 && v13 != 4 )
          {
            if ( v13 == 5 )
              *(_QWORD *)(a1 + 408) += v14;
            goto LABEL_14;
          }
          *(_QWORD *)(a1 + 400) += v14;
        }
LABEL_15:
        v15 = a2[18];
        result = a2 + 16;
        v32 = a2 + 16;
        if ( a2 + 16 != (__int64 *)v15 )
        {
          do
          {
            v16 = sub_26E3C30(i, *(_QWORD *)(v15 + 32) % v37[2], (_DWORD *)(v15 + 32), *(_QWORD *)(v15 + 32));
            if ( v16 && (v17 = *(_QWORD *)v16) != 0 )
            {
              v18 = *(_QWORD *)(v15 + 64);
              v33 = *(_DWORD *)(v17 + 16);
              v19 = v15 + 48;
              v34 = v33 == 6 || ((v33 - 2) & 0xFFFFFFFD) == 0;
              if ( v15 + 48 == v18 )
              {
                v20 = *(_QWORD *)(v15 + 64);
                v21 = 0;
                goto LABEL_21;
              }
            }
            else
            {
              v18 = *(_QWORD *)(v15 + 64);
              v19 = v15 + 48;
              v34 = 0;
              v33 = 0;
              if ( v18 == v15 + 48 )
                goto LABEL_23;
            }
            v20 = v18;
            v21 = 0;
            do
            {
              v21 += *(_QWORD *)(v20 + 104);
              v20 = sub_220EF30(v20);
            }
            while ( v20 != v19 );
LABEL_21:
            if ( v34 )
            {
              *(_QWORD *)(a1 + 400) += v21;
            }
            else if ( v33 == 5 )
            {
              *(_QWORD *)(a1 + 408) += v21;
              v18 = *(_QWORD *)(v15 + 64);
              if ( v20 != v18 )
              {
                do
                {
LABEL_31:
                  v35 = v20;
                  sub_26E3EF0(a1, v18 + 48);
                  v22 = sub_220EF30(v18);
                  v20 = v35;
                  v18 = v22;
                }
                while ( v22 != v35 );
              }
            }
            else if ( v20 != v18 )
            {
              goto LABEL_31;
            }
LABEL_23:
            result = (_QWORD *)sub_220EF30(v15);
            v15 = (__int64)result;
          }
          while ( v32 != result );
        }
      }
    }
  }
  return result;
}

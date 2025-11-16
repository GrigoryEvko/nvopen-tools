// Function: sub_C47810
// Address: 0xc47810
//
__int64 __fastcall sub_C47810(__int64 a1, __int64 a2, char *a3, unsigned __int64 a4, unsigned __int8 a5)
{
  __int64 v5; // r14
  unsigned __int8 v8; // cl
  char *v9; // r15
  __int64 result; // rax
  unsigned int v11; // r13d
  char *v12; // rbx
  __int64 v13; // rax
  char *v14; // r14
  bool v15; // r15
  char *v16; // r12
  __int64 v17; // rbx
  __int64 v18; // r8
  unsigned int v19; // esi
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  int v22; // eax
  unsigned int v23; // ecx
  unsigned __int64 v24; // r13
  size_t v25; // rdx
  __int64 v26; // rdi
  unsigned int v27; // edx
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  unsigned __int8 v32; // [rsp+10h] [rbp-50h]
  char v33; // [rsp+1Bh] [rbp-45h]
  unsigned int v34; // [rsp+1Ch] [rbp-44h]
  unsigned __int8 v35; // [rsp+1Ch] [rbp-44h]
  unsigned int v36; // [rsp+20h] [rbp-40h]
  unsigned __int64 v37; // [rsp+28h] [rbp-38h]

  v5 = a1;
  v8 = a5;
  v33 = *a3;
  if ( ((*a3 - 43) & 0xFD) != 0 )
  {
    result = *(unsigned int *)(a1 + 8);
    v37 = a4;
    v9 = a3;
    if ( (unsigned int)result <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    v9 = a3 + 1;
    v37 = a4 - 1;
    result = *(unsigned int *)(a1 + 8);
    if ( (unsigned int)result <= 0x40 )
    {
LABEL_3:
      *(_QWORD *)a1 = 0;
      goto LABEL_4;
    }
  }
  v32 = a5;
  v35 = a5;
  v24 = ((unsigned __int64)(unsigned int)result + 63) >> 6;
  result = sub_2207820(8 * v24);
  v25 = 8 * v24;
  v8 = v35;
  a5 = v32;
  v26 = result;
  if ( result )
  {
    if ( (__int64)(v24 - 2) < -1 )
      v25 = 8;
    result = (__int64)memset((void *)result, 0, v25);
    v8 = v35;
    a5 = v32;
    v26 = result;
  }
  *(_QWORD *)v5 = v26;
LABEL_4:
  v11 = 4;
  if ( a5 != 16 )
  {
    v11 = 3;
    if ( a5 != 8 )
      v11 = a5 == 2;
  }
  v12 = &a3[a4];
  if ( v9 != v12 )
  {
    v34 = v8;
    v36 = v8 - 11;
    v29 = v8;
    v13 = v5;
    v14 = v9;
    v15 = a5 == 16 || a5 == 36;
    v16 = v12;
    v17 = v13;
    do
    {
      v22 = *v14;
      v23 = v22 - 48;
      if ( v15 )
      {
        v18 = v23;
        if ( v23 > 9 )
        {
          v18 = (unsigned int)(v22 - 55);
          if ( v22 - 65 > v36 )
          {
            v18 = (unsigned int)(v22 - 87);
            if ( v36 < v22 - 97 )
              v18 = 0xFFFFFFFFLL;
          }
        }
      }
      else
      {
        v18 = v23;
        if ( v34 <= v23 )
          v18 = 0xFFFFFFFFLL;
      }
      if ( v37 > 1 )
      {
        if ( v11 )
        {
          v19 = *(_DWORD *)(v17 + 8);
          if ( v19 > 0x40 )
          {
            v31 = v18;
            sub_C47690((__int64 *)v17, v11);
            v18 = v31;
          }
          else
          {
            v20 = 0;
            if ( v11 != v19 )
              v20 = *(_QWORD *)v17 << v11;
            v21 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v19) & v20;
            if ( !v19 )
              v21 = 0;
            *(_QWORD *)v17 = v21;
          }
        }
        else
        {
          v30 = v18;
          sub_C47170(v17, v29);
          v18 = v30;
        }
      }
      ++v14;
      result = sub_C46A40(v17, v18);
    }
    while ( v14 != v16 );
    v5 = v17;
  }
  if ( v33 == 45 )
  {
    v27 = *(_DWORD *)(v5 + 8);
    if ( v27 <= 0x40 )
    {
      v28 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v27) & ~*(_QWORD *)v5;
      if ( !v27 )
        v28 = 0;
      *(_QWORD *)v5 = v28;
    }
    else
    {
      sub_C43D10(v5);
    }
    return sub_C46250(v5);
  }
  return result;
}

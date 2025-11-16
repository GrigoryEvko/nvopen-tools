// Function: sub_37F6040
// Address: 0x37f6040
//
__int64 __fastcall sub_37F6040(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  int v8; // ecx
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // r8d
  __int64 *v12; // rdx
  __int64 v13; // r9
  __int64 *v14; // rdx
  __int64 v15; // r10
  int v16; // edx
  __int64 v17; // rax
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 *v20; // r13
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rdx
  __int64 *v24; // rsi
  __int64 *v25; // rax
  __int64 v26; // rcx
  __int64 *v27; // rdi
  __int64 *v28; // rcx
  int v29; // edx
  int v30; // edx
  int v31; // r11d
  int v32; // r10d
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v36; // [rsp+18h] [rbp-58h]
  __int64 v37; // [rsp+20h] [rbp-50h]
  int v38; // [rsp+28h] [rbp-48h]
  char v39; // [rsp+2Ch] [rbp-44h]
  char v40; // [rsp+30h] [rbp-40h] BYREF

  result = sub_37F5AA0(a1, a2, a3);
  if ( !result )
    goto LABEL_8;
  v8 = *(_DWORD *)(a1 + 488);
  v9 = *(_QWORD *)(a1 + 472);
  if ( !v8 )
    goto LABEL_8;
  v10 = v8 - 1;
  v11 = v10 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( result == *v12 )
  {
LABEL_4:
    v7 = *((unsigned int *)v12 + 2);
  }
  else
  {
    v29 = 1;
    while ( v13 != -4096 )
    {
      v32 = v29 + 1;
      v11 = v10 & (v29 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( result == *v12 )
        goto LABEL_4;
      v29 = v32;
    }
    v7 = 0;
  }
  v6 = v10 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = (__int64 *)(v9 + 16 * v6);
  v15 = *v14;
  if ( a2 == *v14 )
  {
LABEL_6:
    v16 = *((_DWORD *)v14 + 2);
  }
  else
  {
    v30 = 1;
    while ( v15 != -4096 )
    {
      v31 = v30 + 1;
      v6 = v10 & (unsigned int)(v30 + v6);
      v14 = (__int64 *)(v9 + 16LL * (unsigned int)v6);
      v15 = *v14;
      if ( a2 == *v14 )
        goto LABEL_6;
      v30 = v31;
    }
    v16 = 0;
  }
  if ( v16 <= (int)v7 )
  {
LABEL_8:
    v35 = 0;
    v36 = (__int64 *)&v40;
    v17 = *(_QWORD *)(a2 + 24);
    v37 = 2;
    v18 = *(__int64 **)(v17 + 64);
    v33 = v17;
    v19 = *(unsigned int *)(v17 + 72);
    v38 = 0;
    v39 = 1;
    v20 = &v18[v19];
    if ( v18 == v20 )
      return 0;
    do
    {
      v21 = *v18++;
      sub_37F5FF0(a1, v21, a3, (__int64)&v35, v6, v7);
    }
    while ( v20 != v18 );
    v22 = HIDWORD(v37);
    if ( HIDWORD(v37) - v38 == 1 )
    {
      v23 = v36;
      if ( !v39 )
        v22 = (unsigned int)v37;
      v24 = &v36[v22];
      if ( v36 == v24 )
      {
        result = *v36;
        if ( v33 != *(_QWORD *)(*v36 + 24) )
        {
LABEL_12:
          if ( !v39 )
          {
            v34 = result;
            _libc_free((unsigned __int64)v36);
            return v34;
          }
          return result;
        }
      }
      else
      {
        v25 = v36;
        while ( 1 )
        {
          v26 = *v25;
          v27 = v25;
          if ( (unsigned __int64)*v25 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v24 == ++v25 )
          {
            v26 = v27[1];
            break;
          }
        }
        if ( v33 != *(_QWORD *)(v26 + 24) )
        {
          while ( 1 )
          {
            result = *v23;
            v28 = v23;
            if ( (unsigned __int64)*v23 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_12;
            if ( v24 == ++v23 )
            {
              result = v28[1];
              goto LABEL_12;
            }
          }
        }
      }
    }
    result = 0;
    goto LABEL_12;
  }
  return result;
}

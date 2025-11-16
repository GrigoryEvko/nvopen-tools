// Function: sub_15C46E0
// Address: 0x15c46e0
//
__int64 __fastcall sub_15C46E0(_QWORD *a1, __int64 a2, char a3)
{
  __int64 v5; // rdx
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // rsi
  __int64 v8; // rdx
  int v9; // edx
  unsigned __int64 *v10; // r12
  unsigned __int64 v11; // rax
  char *v12; // r12
  int v13; // eax
  char *v14; // r10
  char *v15; // rax
  size_t v16; // r11
  unsigned __int64 v17; // r12
  __int64 *v18; // rdi
  __int64 v20; // rax
  size_t v22; // [rsp+10h] [rbp-60h]
  char *v23; // [rsp+18h] [rbp-58h]
  char *v24; // [rsp+20h] [rbp-50h]
  void *src; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 *v26; // [rsp+38h] [rbp-38h] BYREF

  v5 = *(unsigned int *)(a2 + 8);
  v6 = (unsigned __int64 *)a1[3];
  v7 = (unsigned __int64 *)a1[4];
  v26 = v6;
  if ( !(_DWORD)v5 )
  {
    a3 = 0;
    if ( v7 == v6 )
      goto LABEL_14;
    goto LABEL_7;
  }
  if ( v7 != v6 )
  {
    do
    {
LABEL_7:
      src = v26;
      if ( a3 )
      {
        v11 = *v6;
        if ( v11 == 159 )
        {
          a3 = 0;
        }
        else if ( v11 == 4096 )
        {
          v20 = *(unsigned int *)(a2 + 8);
          if ( (unsigned int)v20 >= *(_DWORD *)(a2 + 12) )
          {
            sub_16CD150(a2, a2 + 16, 0, 8);
            v20 = *(unsigned int *)(a2 + 8);
          }
          a3 = 0;
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v20) = 159;
          ++*(_DWORD *)(a2 + 8);
        }
      }
      v12 = (char *)src;
      v13 = sub_15B11B0((unsigned __int64 **)&src);
      v14 = (char *)src;
      v8 = *(unsigned int *)(a2 + 8);
      v15 = &v12[8 * v13];
      v16 = v15 - (_BYTE *)src;
      v17 = (v15 - (_BYTE *)src) >> 3;
      if ( v17 > (unsigned __int64)*(unsigned int *)(a2 + 12) - v8 )
      {
        v22 = v15 - (_BYTE *)src;
        v23 = (char *)src;
        v24 = v15;
        sub_16CD150(a2, a2 + 16, v17 + v8, 8);
        v8 = *(unsigned int *)(a2 + 8);
        v16 = v22;
        v14 = v23;
        v15 = v24;
      }
      if ( v15 != v14 )
      {
        memcpy((void *)(*(_QWORD *)a2 + 8 * v8), v14, v16);
        LODWORD(v8) = *(_DWORD *)(a2 + 8);
      }
      v9 = v17 + v8;
      v10 = v26;
      *(_DWORD *)(a2 + 8) = v9;
      v6 = &v10[(unsigned int)sub_15B11B0(&v26)];
      v26 = v6;
    }
    while ( v6 != v7 );
    v5 = *(unsigned int *)(a2 + 8);
  }
  if ( a3 )
  {
    if ( *(_DWORD *)(a2 + 12) <= (unsigned int)v5 )
    {
      sub_16CD150(a2, a2 + 16, 0, 8);
      v5 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v5) = 159;
    v5 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v5;
  }
LABEL_14:
  v18 = (__int64 *)(a1[2] & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a1[2] & 4) != 0 )
    v18 = (__int64 *)*v18;
  return sub_15C4420(v18, *(void **)a2, v5, 0, 1);
}

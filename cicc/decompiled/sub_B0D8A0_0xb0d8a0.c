// Function: sub_B0D8A0
// Address: 0xb0d8a0
//
__int64 __fastcall sub_B0D8A0(_QWORD *a1, __int64 a2, char a3, char a4)
{
  __int64 v5; // rdx
  unsigned __int64 *v6; // rcx
  unsigned __int64 *v7; // rax
  __int64 v8; // rdx
  int v9; // r9d
  unsigned __int64 *v10; // r12
  unsigned __int64 v11; // rax
  char *v12; // r12
  int v13; // eax
  char *v14; // r10
  char *v15; // rax
  size_t v16; // r11
  __int64 v17; // r12
  __int64 *v18; // rsi
  __int64 *v19; // rdi
  __int64 v21; // rax
  size_t v23; // [rsp+10h] [rbp-60h]
  char *v24; // [rsp+18h] [rbp-58h]
  char *v25; // [rsp+20h] [rbp-50h]
  unsigned __int64 *v26; // [rsp+28h] [rbp-48h]
  void *src; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 *v28; // [rsp+38h] [rbp-38h] BYREF

  if ( a4 )
  {
    sub_A188E0(a2, 4099);
    sub_A188E0(a2, 1);
  }
  v5 = *(unsigned int *)(a2 + 8);
  v6 = (unsigned __int64 *)a1[3];
  v7 = (unsigned __int64 *)a1[2];
  v26 = v6;
  v28 = v7;
  if ( !(_DWORD)v5 )
  {
    a3 = 0;
    if ( v6 == v7 )
      goto LABEL_18;
    goto LABEL_9;
  }
  if ( v6 != v7 )
  {
    do
    {
LABEL_9:
      src = v28;
      if ( a3 )
      {
        v11 = *v7;
        if ( v11 == 159 )
        {
          a3 = 0;
        }
        else if ( v11 == 4096 )
        {
          v21 = *(unsigned int *)(a2 + 8);
          if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
          {
            sub_C8D5F0(a2, a2 + 16, v21 + 1, 8);
            v21 = *(unsigned int *)(a2 + 8);
          }
          a3 = 0;
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v21) = 159;
          ++*(_DWORD *)(a2 + 8);
        }
      }
      v12 = (char *)src;
      v13 = sub_AF4160((unsigned __int64 **)&src);
      v14 = (char *)src;
      v8 = *(unsigned int *)(a2 + 8);
      v15 = &v12[8 * v13];
      v16 = v15 - (_BYTE *)src;
      v17 = (v15 - (_BYTE *)src) >> 3;
      if ( v17 + v8 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v23 = v15 - (_BYTE *)src;
        v24 = (char *)src;
        v25 = v15;
        sub_C8D5F0(a2, a2 + 16, v17 + v8, 8);
        v8 = *(unsigned int *)(a2 + 8);
        v16 = v23;
        v14 = v24;
        v15 = v25;
      }
      if ( v15 != v14 )
      {
        memcpy((void *)(*(_QWORD *)a2 + 8 * v8), v14, v16);
        LODWORD(v8) = *(_DWORD *)(a2 + 8);
      }
      v9 = v17 + v8;
      v10 = v28;
      *(_DWORD *)(a2 + 8) = v9;
      v7 = &v10[(unsigned int)sub_AF4160(&v28)];
      v28 = v7;
    }
    while ( v7 != v26 );
  }
  if ( a3 )
    sub_A188E0(a2, 159);
  v5 = *(unsigned int *)(a2 + 8);
LABEL_18:
  v18 = *(__int64 **)a2;
  v19 = (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a1[1] & 4) != 0 )
    v19 = (__int64 *)*v19;
  return sub_B0D000(v19, v18, v5, 0, 1);
}

// Function: sub_2F56EC0
// Address: 0x2f56ec0
//
__int64 __fastcall sub_2F56EC0(__int64 a1, char **a2, __int64 a3)
{
  int v6; // r14d
  unsigned __int64 v7; // rcx
  __int64 v8; // r13
  unsigned __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // r13
  _DWORD *v13; // rcx
  int v14; // eax
  char *v15; // rsi
  char *v16; // rsi
  unsigned __int64 v18; // rdi
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r10
  _DWORD *v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // r8
  int v25; // eax
  __int64 v26; // r9
  _DWORD *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rcx
  int v30; // [rsp+4h] [rbp-5Ch]
  __int64 v31; // [rsp+8h] [rbp-58h]
  int v32; // [rsp+8h] [rbp-58h]
  int v33; // [rsp+10h] [rbp-50h]
  int v34; // [rsp+14h] [rbp-4Ch]
  __int64 v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a3 + 112);
  v7 = *(unsigned int *)(a1 + 928);
  v8 = v6 & 0x7FFFFFFF;
  v9 = (unsigned int)(v8 + 1);
  v10 = v9;
  if ( (unsigned int)v9 > (unsigned int)v7 && v9 != v7 )
  {
    if ( v9 >= v7 )
    {
      v19 = *(unsigned int *)(a1 + 936);
      v20 = *(unsigned int *)(a1 + 940);
      v21 = v9 - v7;
      if ( v9 > *(unsigned int *)(a1 + 932) )
      {
        v30 = *(_DWORD *)(a1 + 940);
        v33 = *(_DWORD *)(a1 + 936);
        v31 = v9 - v7;
        sub_C8D5F0(a1 + 920, (const void *)(a1 + 936), v9, 8u, v19, v20);
        v7 = *(unsigned int *)(a1 + 928);
        LODWORD(v20) = v30;
        LODWORD(v19) = v33;
        v21 = v31;
        v10 = v8 + 1;
        v9 = (unsigned int)(v8 + 1);
      }
      v22 = (_DWORD *)(*(_QWORD *)(a1 + 920) + 8 * v7);
      v23 = v21;
      do
      {
        if ( v22 )
        {
          *v22 = v19;
          v22[1] = v20;
        }
        v22 += 2;
        --v23;
      }
      while ( v23 );
      *(_DWORD *)(a1 + 928) += v21;
    }
    else
    {
      *(_DWORD *)(a1 + 928) = v9;
    }
  }
  v11 = *(_QWORD *)(a1 + 920);
  v12 = 8 * v8;
  v13 = (_DWORD *)(v11 + v12);
  if ( !*(_DWORD *)(v11 + v12) )
  {
    v18 = *(unsigned int *)(a1 + 928);
    if ( v10 > (unsigned int)v18 && v9 != v18 )
    {
      if ( v9 >= v18 )
      {
        v24 = *(unsigned int *)(a1 + 936);
        v25 = *(_DWORD *)(a1 + 940);
        v26 = v9 - v18;
        if ( v9 > *(unsigned int *)(a1 + 932) )
        {
          v32 = *(_DWORD *)(a1 + 936);
          v34 = *(_DWORD *)(a1 + 940);
          v35 = v9 - v18;
          sub_C8D5F0(a1 + 920, (const void *)(a1 + 936), v9, 8u, v24, v26);
          v11 = *(_QWORD *)(a1 + 920);
          v18 = *(unsigned int *)(a1 + 928);
          LODWORD(v24) = v32;
          v25 = v34;
          v26 = v35;
        }
        v27 = (_DWORD *)(v11 + 8 * v18);
        v28 = v26;
        do
        {
          if ( v27 )
          {
            *v27 = v24;
            v27[1] = v25;
          }
          v27 += 2;
          --v28;
        }
        while ( v28 );
        v29 = *(_QWORD *)(a1 + 920);
        *(_DWORD *)(a1 + 928) += v26;
        v13 = (_DWORD *)(v12 + v29);
      }
      else
      {
        *(_DWORD *)(a1 + 928) = v10;
      }
    }
    *v13 = 1;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 976) + 16LL))(*(_QWORD *)(a1 + 976), a3);
  v15 = a2[1];
  LODWORD(v36) = v14;
  HIDWORD(v36) = ~v6;
  if ( v15 == a2[2] )
  {
    sub_1E0C2B0(a2, v15, &v36);
    v16 = a2[1];
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v36;
      v15 = a2[1];
    }
    v16 = v15 + 8;
    a2[1] = v16;
  }
  return sub_2F4CA50((__int64)*a2, ((v16 - *a2) >> 3) - 1, 0, *((_QWORD *)v16 - 1));
}

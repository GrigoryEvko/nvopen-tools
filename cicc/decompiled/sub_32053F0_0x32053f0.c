// Function: sub_32053F0
// Address: 0x32053f0
//
void __fastcall sub_32053F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 *v7; // rbx
  int v8; // ecx
  int v9; // edx
  __int64 *v10; // r14
  unsigned __int64 v11; // r10
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __int64 *v14; // rdx
  __int64 *v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi
  __int64 v18; // rdx
  const void *v19; // r11
  int v20; // eax
  __int64 *v21; // rsi
  int v22; // [rsp+8h] [rbp-88h]
  __int64 v23; // [rsp+18h] [rbp-78h]
  __int64 v24; // [rsp+18h] [rbp-78h]
  int v25; // [rsp+20h] [rbp-70h]
  int v26; // [rsp+20h] [rbp-70h]
  int v27; // [rsp+28h] [rbp-68h]
  __int64 v28; // [rsp+28h] [rbp-68h]
  __int64 *v29; // [rsp+30h] [rbp-60h] BYREF
  __int64 v30; // [rsp+38h] [rbp-58h]
  _BYTE v31[80]; // [rsp+40h] [rbp-50h] BYREF

  v30 = 0x400000000LL;
  v6 = *(unsigned int *)(a1 + 1288);
  v29 = (__int64 *)v31;
  if ( (_DWORD)v6 )
  {
    do
    {
      v7 = *(__int64 **)(a1 + 1280);
      v8 = *(_DWORD *)(a1 + 1292);
      if ( v7 == (__int64 *)(a1 + 1296) || v29 == (__int64 *)v31 )
      {
        v11 = (unsigned int)v6;
        if ( HIDWORD(v30) < (unsigned int)v6 )
        {
          sub_C8D5F0((__int64)&v29, v31, (unsigned int)v6, 8u, a5, a6);
          v11 = *(unsigned int *)(a1 + 1288);
          LODWORD(v6) = *(_DWORD *)(a1 + 1288);
        }
        v12 = (unsigned int)v30;
        a6 = v11;
        if ( (unsigned int)v30 <= v11 )
          a6 = (unsigned int)v30;
        if ( a6 )
        {
          a5 = 8 * a6;
          v13 = 0;
          do
          {
            v14 = &v29[v13 / 8];
            v15 = (__int64 *)(v13 + *(_QWORD *)(a1 + 1280));
            v13 += 8LL;
            v16 = *v15;
            *v15 = *v14;
            *v14 = v16;
          }
          while ( a5 != v13 );
          v11 = *(unsigned int *)(a1 + 1288);
          v12 = (unsigned int)v30;
          LODWORD(v6) = *(_DWORD *)(a1 + 1288);
        }
        v7 = v29;
        v10 = &v29[v12];
        if ( v11 <= v12 )
        {
          if ( v11 < v12 )
          {
            v20 = v11;
            v21 = &v29[a6];
            if ( v21 != v10 )
            {
              v22 = v12;
              v24 = a6;
              v26 = v11;
              v28 = a6;
              memcpy((void *)(*(_QWORD *)(a1 + 1280) + 8 * v11), v21, 8 * v12 - 8 * a6);
              v7 = v29;
              v20 = *(_DWORD *)(a1 + 1288);
              LODWORD(v12) = v22;
              a6 = v24;
              LODWORD(v11) = v26;
              v10 = &v29[v28];
            }
            LODWORD(v30) = a6;
            *(_DWORD *)(a1 + 1288) = v20 + v12 - v11;
          }
        }
        else
        {
          v17 = *(_QWORD *)(a1 + 1280);
          v18 = v11;
          v25 = v12;
          v27 = v11;
          v19 = (const void *)(v17 + 8 * a6);
          if ( v19 != (const void *)(8 * v11 + v17) )
          {
            v23 = a6;
            memcpy(&v29[v12], v19, v18 * 8 - 8 * a6);
            v7 = v29;
            a6 = v23;
            LODWORD(v6) = v30 + v27 - v25;
            v18 = (unsigned int)v6;
          }
          LODWORD(v30) = v6;
          v10 = &v7[v18];
          *(_DWORD *)(a1 + 1288) = a6;
        }
      }
      else
      {
        *(_QWORD *)(a1 + 1280) = v29;
        v9 = HIDWORD(v30);
        v10 = &v7[v6];
        v29 = v7;
        *(_DWORD *)(a1 + 1288) = 0;
        LODWORD(v30) = v6;
        *(_DWORD *)(a1 + 1292) = v9;
        HIDWORD(v30) = v8;
      }
      for ( ; v10 != v7; ++v7 )
      {
        if ( *v7 )
          sub_3205010(a1, *v7);
      }
      v6 = *(unsigned int *)(a1 + 1288);
      LODWORD(v30) = 0;
    }
    while ( (_DWORD)v6 );
    if ( v29 != (__int64 *)v31 )
      _libc_free((unsigned __int64)v29);
  }
}

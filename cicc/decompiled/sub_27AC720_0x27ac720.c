// Function: sub_27AC720
// Address: 0x27ac720
//
void __fastcall sub_27AC720(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  int v8; // eax
  __int64 v9; // r8
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rdx
  unsigned __int64 v16; // rbx
  int v17; // eax
  __int64 v18; // rdi
  __int64 v19; // rcx
  int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // [rsp+0h] [rbp-A0h]
  __int64 v23; // [rsp+10h] [rbp-90h]
  __int64 v24; // [rsp+20h] [rbp-80h]
  __int64 v25; // [rsp+28h] [rbp-78h]
  int v26; // [rsp+30h] [rbp-70h]
  char *v27; // [rsp+38h] [rbp-68h] BYREF
  __int64 v28; // [rsp+40h] [rbp-60h]
  _BYTE v29[88]; // [rsp+48h] [rbp-58h] BYREF

  if ( a1 != a2 && a2 != a1 + 72 )
  {
    v7 = a1 + 96;
    v22 = a1 + 24;
    do
    {
      v8 = *(_DWORD *)(v7 - 8);
      v9 = *(unsigned int *)(v7 - 24);
      v10 = v7 - 24;
      v11 = v7;
      v12 = *(unsigned int *)(v7 - 12);
      v24 = *(_QWORD *)(v7 - 24);
      v13 = *(unsigned int *)(v7 + 8);
      v25 = *(_QWORD *)(v7 - 16);
      v26 = v8;
      if ( v8 <= *(_DWORD *)(a1 + 16) )
      {
        v27 = v29;
        v19 = 0x400000000LL;
        v28 = 0x400000000LL;
        if ( (_DWORD)v13 )
        {
          sub_27AC070((__int64)&v27, (char **)v7, v13, 0x400000000LL, v9, a6);
          v8 = v26;
        }
        if ( v8 > *(_DWORD *)(v7 - 80) )
        {
          do
          {
            v20 = *(_DWORD *)(v11 - 96);
            v21 = v11;
            v10 = v11 - 96;
            v11 -= 72;
            *(_DWORD *)(v11 + 48) = v20;
            *(_DWORD *)(v11 + 52) = *(_DWORD *)(v11 - 20);
            *(_DWORD *)(v11 + 56) = *(_DWORD *)(v11 - 16);
            *(_DWORD *)(v11 + 60) = *(_DWORD *)(v11 - 12);
            *(_DWORD *)(v11 + 64) = *(_DWORD *)(v11 - 8);
            sub_27AC070(v21, (char **)v11, v13, v19, v9, a6);
            v8 = v26;
          }
          while ( v26 > *(_DWORD *)(v11 - 80) );
        }
        *(_DWORD *)(v10 + 16) = v8;
        *(_QWORD *)v10 = v24;
        *(_QWORD *)(v10 + 8) = v25;
        sub_27AC070(v11, &v27, v25, v19, v9, a6);
        if ( v27 != v29 )
          _libc_free((unsigned __int64)v27);
        v23 = v7 + 48;
      }
      else
      {
        v27 = v29;
        v28 = 0x400000000LL;
        if ( (_DWORD)v13 )
          sub_27AC070((__int64)&v27, (char **)v7, v13, v12, v9, a6);
        v14 = v7;
        v15 = v10 - a1;
        v23 = v7 + 48;
        v16 = 0x8E38E38E38E38E39LL * ((v10 - a1) >> 3);
        if ( v15 > 0 )
        {
          do
          {
            v17 = *(_DWORD *)(v14 - 96);
            v18 = v14;
            v14 -= 72;
            *(_DWORD *)(v14 + 48) = v17;
            *(_DWORD *)(v14 + 52) = *(_DWORD *)(v14 - 20);
            *(_DWORD *)(v14 + 56) = *(_DWORD *)(v14 - 16);
            *(_DWORD *)(v14 + 60) = *(_DWORD *)(v14 - 12);
            *(_DWORD *)(v14 + 64) = *(_DWORD *)(v14 - 8);
            sub_27AC070(v18, (char **)v14, v15, v12, v9, a6);
            --v16;
          }
          while ( v16 );
        }
        *(_QWORD *)a1 = v24;
        *(_QWORD *)(a1 + 8) = v25;
        *(_DWORD *)(a1 + 16) = v26;
        sub_27AC070(v22, &v27, v15, v12, v9, a6);
        if ( v27 != v29 )
          _libc_free((unsigned __int64)v27);
      }
      v7 += 72;
    }
    while ( a2 != v23 );
  }
}
